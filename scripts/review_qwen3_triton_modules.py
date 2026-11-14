"""Review isolated actual-Qwen3-8B module timings; never adopt from microbenchmarks.

These CUDA-event measurements exclude FSDP/NCCL/optimizer and use local BF16
parameter gradients. They explain complete-step ablations without replacing
their numeric, quality, memory or performance gates. The untimed dispatcher
guard is evidence of guarded VJP execution, not a separately captured trace.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math,statistics

ROOT=Path(__file__).resolve().parents[1]
import argparse
D=C=None

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*2**20),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    global D,C
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    C=parser.parse_args().output;D=C/'measurements'
    manifest=json.loads((C/'manifest.json').read_text())
    result=json.loads((D/'result.json').read_text())
    monitor=json.loads((C/'gpu.json').read_text())
    assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and monitor['observation_errors']==0
    assert result['complete'] and result['model']=='Qwen/Qwen3-8B'
    assert result['math_policy']==dict(compute='BF16',TF32=False,allow_bf16_reduced_precision_reduction=False,deterministic_fill=True)
    import torch
    torch.set_num_threads(2)
    proof=json.loads((D/'reference_capture.json').read_text())
    assert proof['complete'] and proof['model']=='Qwen/Qwen3-8B' and proof['layers']==36 and proof['parameter_tensors']==399
    assert proof['input_manifest_sha256']==result['input_manifest_sha256']
    assert sha(D/'reference_capture.pt')==proof['capture_sha256']==result['reference_capture_sha256']
    captured=torch.load(D/'reference_capture.pt',map_location='cpu',weights_only=True,mmap=True)
    assert captured['ids'].shape==(1,512) and captured['ids'].dtype==torch.int64
    assert torch.equal(captured['ids'],captured['reference']['model.embed_tokens']['input'])
    assert captured['refs']['logits'].shape==(1,512,151936) and captured['refs']['logits'].dtype==torch.bfloat16
    assert torch.isfinite(captured['refs']['logits']).all() and torch.isfinite(captured['refs']['loss'])
    assert len(captured['reference'])==len(proof['captured_modules'])==14
    for name,entry in captured['reference'].items():
        value=entry['input']
        if name=='model.embed_tokens':continue
        assert value.dtype==torch.bfloat16 and torch.isfinite(value).all()
        if name.endswith('q_norm'):assert value.shape==(1,512,32,128)
        elif name.endswith('k_norm'):assert value.shape==(1,512,8,128)
        else:assert value.shape==(1,512,4096)
        if name.endswith('self_attn'):
            assert len(entry['positions'])==2
            assert all(t.shape==(1,512,128) and t.dtype==torch.bfloat16 and torch.isfinite(t).all() for t in entry['positions'])
    del captured

    for mapping in (manifest['sources'],result['sources']):
        for path,h in mapping.items():assert sha(path)==h,(path,'source hash changed')
    from training.provenance import source_snapshot
    assert source_snapshot(ROOT)==manifest['core_snapshot']
    variants={
        'attention':['hf','triton_base','triton_optimized','ablate_attention','ablate_norm_rope','ablate_linear'],
        'norm_rope':['hf','triton_base','triton_optimized'],
        'mlp':['hf','triton_base','triton_optimized','ablate_mlp','ablate_linear'],
        'lm_head':['hf','triton_base','triton_optimized'],
        'loss':['hf','triton_base','triton_optimized'],
    }
    counts={'attention':41943296,'norm_rope':256,'mlp':150994944,'lm_head':622329856,'loss':0}
    expected=[(layer,kind,variant,order) for layer in (0,17,35) for kind in ('attention','norm_rope','mlp') for order in (1,2) for variant in (variants[kind] if order==1 else list(reversed(variants[kind])))]
    expected += [(None,kind,variant,order) for kind in ('lm_head','loss') for order in (1,2) for variant in (variants[kind] if order==1 else list(reversed(variants[kind])))]
    rows=result['rows'];assert len(rows)==96
    assert [(r['layer'],r['module'],r['variant'],r['order']) for r in rows]==expected
    indexed={};summary=[]
    for row in rows:
        kind=row['module'];assert row['parameter_count']==counts[kind]
        shapes=row['input_shapes']
        if kind=='norm_rope':assert shapes==[[1,512,32,128],[1,512,8,128]]
        elif kind=='loss':assert shapes==[[1,512,151936]]
        else:assert shapes==[[1,512,4096]]
        assert row['untimed_no_fallback_VJP_finite']
        for part in ('forward','backward'):
            samples=row[part+'_ms'];assert len(samples)==20 and all(math.isfinite(t) and t>0 for t in samples)
            assert row['median_'+part+'_ms']==statistics.median(samples)
        assert math.isfinite(row['peak_extra_allocated_gib']) and row['peak_extra_allocated_gib']>=0
        assert 0<row['peak_reserved_gib']<=41.875
        indexed[(row['layer'],kind,row['variant'],row['order'])]=row
    for layer,kind,variant,order in expected:
        row=indexed[layer,kind,variant,order]
        hf=indexed[layer,kind,'hf',order];base=indexed[layer,kind,'triton_base',order]
        total=lambda r:r['median_forward_ms']+r['median_backward_ms']
        summary.append(dict(layer=layer,module=kind,variant=variant,order=order,
            median_forward_ms=row['median_forward_ms'],median_backward_ms=row['median_backward_ms'],
            sum_of_separate_medians_ms=total(row),
            forward_reduction_vs_HF=1-row['median_forward_ms']/hf['median_forward_ms'],
            backward_reduction_vs_HF=1-row['median_backward_ms']/hf['median_backward_ms'],
            fwd_bwd_reduction_vs_HF=1-total(row)/total(hf),
            fwd_bwd_reduction_vs_Triton_base=1-total(row)/total(base),
            peak_extra_allocated_gib=row['peak_extra_allocated_gib'],peak_reserved_gib=row['peak_reserved_gib']))
    ablations=[]
    applicable={'attention':['ablate_attention','ablate_norm_rope','ablate_linear'],'mlp':['ablate_mlp','ablate_linear']}
    for layer in (0,17,35):
        for kind,names in applicable.items():
            for variant in names:
                orders=[]
                for order in (1,2):
                    opt=indexed[layer,kind,'triton_optimized',order];abl=indexed[layer,kind,variant,order]
                    orders.append(dict(order=order,forward_increase_fraction=abl['median_forward_ms']/opt['median_forward_ms']-1,
                        backward_increase_fraction=abl['median_backward_ms']/opt['median_backward_ms']-1,
                        fwd_bwd_increase_fraction=(abl['median_forward_ms']+abl['median_backward_ms'])/(opt['median_forward_ms']+opt['median_backward_ms'])-1))
                ablations.append(dict(layer=layer,module=kind,variant=variant,orders=orders))
    report=dict(complete=True,audit_passed=True,model='Qwen/Qwen3-8B',case_count=len(rows),
        warmup_per_case=5,timing_samples_per_part=20,orders=2,
        summary=summary,ablations=ablations,performance_adopted=False,scope=__doc__,
        sources={str(p):sha(p) for p in (Path(__file__),C/'manifest.json',C/'gpu.json',D/'result.json')})
    (D/'independent_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,audit_passed=True,cases=len(rows),performance_adopted=False)),flush=True)

if __name__=='__main__':main()
