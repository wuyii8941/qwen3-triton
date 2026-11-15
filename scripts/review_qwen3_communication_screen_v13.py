"""Independent CPU audit of the nine-update communication/allocation screen.

This validates short-screen evidence only. It cannot establish original399
reference-gradient equivalence, full validation, convergence or adoption.
"""
import argparse,json,statistics
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_vocab_v12_contract import audit_trace
from training.provenance import source_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve()
    manifest=json.loads((out/'manifest.json').read_text());result=json.loads((out/'result.json').read_text())
    history=json.loads((out/'attempts.json').read_text())
    assert result['complete'] and result['full_training_qualified'] is False
    for path,digest in manifest['sources'].items():assert sha(path)==digest,path
    assert source_snapshot(root)==manifest['core_snapshot']
    roles=manifest['roles'];assert [r['name'] for r in roles]==['control_first','separate','compute','buffers','compute_buffers','control_last']
    plan=json.loads((root/'results/qwen3_triton_v10_gemm_plan_001.json').read_text())
    summaries={};inputs=None;losses=None;raw={}
    for name in ['overwrite_check',*[r['name'] for r in roles]]:
        attempts=[r for r in history if r['role']==name];assert attempts
        clean=[r for r in attempts if not r['resource_contaminated']]
        assert len(clean)==1 and clean[0]['returncode']==0 and clean[0] is attempts[-1]
        folder=out/clean[0]['folder'];assert (out/name).resolve()==folder.resolve()
        for row in attempts:
            d=out/row['folder'];assert sha(d/'monitor.json')==row['monitor_sha256'] and sha(d/'worker.log')==row['log_sha256']
        monitor=json.loads((folder/'monitor.json').read_text());owned=set(monitor['known_owned_pids'])
        assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors'] and monitor['samples']
        assert all(not s.get('foreign_pids') and 'observation_error' not in s and all(p['pid'] in owned for p in s['apps']) for s in monitor['samples'])
        raw[str(folder/'monitor.json')]=sha(folder/'monitor.json')
        if name=='overwrite_check':
            check=json.loads((folder/'check.json').read_text());assert check['complete'] and check['deterministic_fill_still_enabled'] and len(check['rows'])==6
            for r in check['rows']:
                fields=[v for k,v in r.items() if k.endswith('bitwise_equal')];assert len(fields)==1 and all(fields[0]) and r['poisoned']
            raw[str(folder/'check.json')]=sha(folder/'check.json');continue
        role=next(r for r in roles if r['name']==name)
        d=json.loads((folder/'run/result.json').read_text());assert d['complete'] and d['screen_only'] and d['model']=='Qwen/Qwen3-8B' and d['layers']==36 and d['world_size']==4
        for path,digest in d['sources'].items():assert sha(path)==digest,path
        assert d['math_policy']==dict(compute='BF16',master_gradient_moments='FP32',TF32=False,allow_bf16_reduced_precision_reduction=False)
        if inputs is None:inputs=d['input_manifest_sha256']
        assert d['input_manifest_sha256']==inputs
        samples=[r['wall_seconds'] for r in d['ranks'][0]['rows'] if r['measured']];assert len(samples)==4
        own_losses=[r['global_loss'] for r in d['ranks'][0]['rows']]
        if losses is None:losses=own_losses
        assert own_losses==losses
        for rank,row in enumerate(d['ranks']):
            assert row['rank']==rank and row['updates']==9 and row['parameter_count']==row['optimizer_state_count']==399
            assert row['all_master_grad_moment_finite_fp32'] and max(row['measured_training_peak'].values())<=41.875
            assert row['gradient_comparisons']==[] and row['full_validation']==[]
            assert [r['global_loss'] for r in row['rows']]==losses and [r['wall_seconds'] for r in row['rows'] if r['measured']]==samples
            alloc=row['overwrite_outputs'];assert alloc['enabled']==role['compute'] and alloc['deterministic_algorithms_enabled'] and alloc['deterministic_fill_enabled']
            assert (alloc['calls']>0)==role['compute']
            comm=row['communication'];assert comm['mode']==role['communication']
            if role['buffers']:assert comm['overwrite_communication_buffers'] and comm['AG_calls']>0 and comm['RS_calls']>0
        trace=audit_trace(folder/'run/training_trace.json','triton_optimized',True,plan,fused_dw=True,dw_config=(64,128,32,8,8,3))
        assert trace['passed'],trace
        row=result['rows'][name];assert row['median']==statistics.median(samples) and row['samples']==samples
        summaries[name]=dict(median=row['median'],samples=samples,losses_equal=True,allocated_GiB=row['allocated_GiB'],reserved_GiB=row['reserved_GiB'],actual_CUDA_contract_passed=True)
        raw[str(folder/'run/result.json')]=sha(folder/'run/result.json');raw[str(folder/'run/training_trace.json')]=sha(folder/'run/training_trace.json')
    candidates=[]
    for role in roles:
        name=role['name'];row=result['rows'][name];s=summaries[name]
        gains=[1-s['median']/summaries[c]['median'] for c in ['control_first','control_last']]
        assert gains==row['reductions_vs_endpoint_controls']
        eligible=name not in ['control_first','control_last'] and min(gains)>=.03
        assert eligible==row['eligible_for_formal']
        s['reductions_vs_endpoint_controls']=gains;s['eligible_for_formal']=eligible
        if eligible:candidates.append(name)
    selected=min(candidates,key=lambda n:summaries[n]['median']) if candidates else None
    assert result['selected']==selected
    report=dict(complete=True,audit_passed=True,full_training_qualified=False,selected=selected,rows=summaries,
        original_gradient_reference_not_checked=True,full_validation_not_run=True,
        first_clean_resource_selection_passed=True,raw_evidence_sha256=raw,
        sources={str(p):sha(p) for p in [Path(__file__),out/'manifest.json',out/'result.json',out/'attempts.json']},scope=__doc__)
    with (out/'short_screen_review.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    with a.summary.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(audit_passed=True,selected=selected,medians={n:r['median'] for n,r in summaries.items()}),indent=2))


if __name__=='__main__':main()
