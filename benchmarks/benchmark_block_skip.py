"""First-stage screening of BLASST-style pruning on REAL Qwen3 decode states.

Fixed threshold sweep, not the paper's calibrated threshold/model recipe.
Compare against BOTH zero-threshold own kernel and native Flash. A win over
our own dense kernel alone does not justify model integration.
"""
import argparse
import json
from pathlib import Path
import statistics
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
from model.hf_inference import qwen3_triton_inference,pack_qkv_weights
from model.static_decode import StaticDecodeRunner,StaticDecodeCache
from ops.triton.block_skip_decode import block_skip_attention
from benchmarks.benchmark_frequency_sparse import ROOT,MODEL,sha,observe,graph_us


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='v1')
    parser.add_argument('--seqs',default='8192,32768')
    args=parser.parse_args()
    target=ROOT/'artifacts'/f'block_skip_{args.tag}.json'
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(4)
    torch.manual_seed(20261101)
    data=json.loads((ROOT/'artifacts/frequency_wikitext_data.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    tokens=torch.tensor(tokenizer(data['splits']['test']['text'],add_special_tokens=False,verbose=False)['input_ids'])
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,
        torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    packed=pack_qkv_weights(model)
    report=dict(manifest=dict(algorithm='BLASST-inspired local running-max skip, reverse tile order per split',
        paper='https://arxiv.org/html/2512.12087v3',date='2026-11-01',gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,dtype='float16',head_dim=128,gqa='16 query heads / 8 KV heads',
        model=str(MODEL),config_sha256=sha(MODEL/'config.json'),data_sha256=data['splits']['test']['sha256'],
        source_sha256={p:sha(ROOT/p) for p in ['ops/triton/block_skip_decode.py','benchmarks/benchmark_block_skip.py']},
        timing='CUDA Graph GPU time, 8 samples x20 replays; no hooks inside timing',
        scope='real activation operator screening only; no end-to-end or task quality claim'),cases=[])
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION),qwen3_triton_inference(model,
            fuse_qk=True,compact_gqa=True,decode_mlp=True,packed_qkv=True,packed_weights=packed,
            prefill_mlp=True,static_decode=True,direct_k_cache=True,direct_v_cache=True,fuse_residual=True):
        for seq in map(int,args.seqs.split(',')):
            if seq+16>model.config.max_position_embeddings:raise ValueError('Context limit')
            ids=tokens[:seq+16].unsqueeze(0).cuda()
            runner=StaticDecodeRunner(model,1,seq+16)
            runner.prefill(ids[:,:seq])
            states={}
            def collect(q,layer,scale,out):
                if layer in (0,7,14,21,27):states[layer]=(q.clone(),out.clone(),scale)
            with observe(runner.cache,collect):
                for pos in range(seq,seq+16):runner.decode(ids[:,pos:pos+1],replay=False)
            case={'seq_len':seq,'layers':{}}
            report['cases'].append(case)
            for layer,(q4,expected,scale) in states.items():
                q=q4[:,:,0]
                k,v=runner.cache.key_cache[layer],runner.cache.value_cache[layer]
                used=runner.cache.used
                rows=[]
                baseline=graph_us(lambda:StaticDecodeCache.attend_decode(runner.cache,q4,layer,scale))
                for splits,bn in ((8,64),(16,64),(32,64),(16,128)):
                    for threshold in (0,1e-6,1e-4,1e-2):
                        fn=lambda:block_skip_attention(q,k,v,used,threshold=threshold,splits=splits,block_n=bn,scale=scale)
                        out,stats=fn()
                        rel=float((out.float()-expected[:,0].float()).norm()/expected.float().norm().clamp_min(1e-12))
                        if threshold==0 and rel>.003:raise AssertionError('Dense variant failed parity')
                        timing=graph_us(fn)
                        row=dict(splits=splits,block_n=bn,threshold=threshold,
                            skipped_fraction=float(stats[...,0].sum()/stats[...,1].sum()),
                            relative_l2=rel,**timing,speedup_vs_flash=baseline['median_us']/timing['median_us'])
                        rows.append(row)
                case['layers'][str(layer)]={'native_flash':baseline,'candidates':rows}
                fastest=min(rows,key=lambda r:r['median_us'])
                print('screen',seq,layer,'flash_us',round(baseline['median_us'],2),'fastest',
                      {k:v for k,v in fastest.items() if k!='samples_us'},flush=True)
                target.write_text(json.dumps(report,indent=2))
            # Screening only: parameters must later be chosen on calibration
            # data and independently quality-validated before any promotion.
            case['any_operator_candidate']=any(r['speedup_vs_flash']>1.1 and r['relative_l2']<.02
                for l in case['layers'].values() for r in l['candidates'])
            del runner,states
        report['decision']='Not enabled. Operator-only screening is not a task-quality or end-to-end acceptance.'
        target.write_text(json.dumps(report,indent=2))
    print('saved',target,flush=True)


if __name__=='__main__':main()
