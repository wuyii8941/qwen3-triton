"""Real Qwen E2E ablation: existing best path, QK fusion, compact GQA, both.

All paths use the same weights/tokens/Flash SDPA and include HF cache updates.
Preparation, JIT and profiler are outside timing. Randomized paired rounds
separate each optimization's contribution; no operator-only speedup claims.
"""
import argparse
from collections import Counter,defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
import torch
import transformers
import triton
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
from model.hf_inference import qwen3_triton_inference


def check(a,b):
    # Preserve the existing 2% gate, but retain failures for diagnosis instead
    # of aborting before observing the candidates. Failed rows are NOT accepted.
    a,b=a.float(),b.float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise AssertionError('Nonfinite output')
    rel=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
    return dict(relative_l2=rel,max_abs=float((a-b).abs().max()),passes_l2_gate=rel<.02)


def bootstrap_speedup(base,candidate):
    ratios=[a/b for a,b in zip(base,candidate)]
    rng=random.Random(1984)
    medians=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(2000))
    return dict(paired_median_speedup=statistics.median(ratios),
                bootstrap_95ci=[medians[50],medians[1949]],
                scope='paired rounds in this run; not cross-day reproducibility')


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dtype',choices=['float16','bfloat16'],default='bfloat16')
    parser.add_argument('--repeats',type=int,default=24)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    if args.repeats<8: raise ValueError('Use at least 8 repeats')
    root=Path(__file__).resolve().parents[1]
    path=Path('/data1/tzh/models/Qwen/Qwen3-1.7B')
    torch.manual_seed(20260929)
    rng=random.Random(20260929)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
        torch_dtype=getattr(torch,args.dtype),attn_implementation='sdpa').cuda().eval()
    def ctx(name):
        return nullcontext() if name=='hf' else qwen3_triton_inference(model,decode_mlp=True,
            fuse_qk=name in ('qk_fused','combined'),compact_gqa=name in ('compact_gqa','combined'))
    def sha(p):
        with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
    sources=['model/hf_inference.py','model/hf_attention.py','ops/triton/hf_qk_norm_rope.py',
             'ops/triton/decode_gate_up.py','benchmarks/benchmark_hf_attention_e2e.py']
    report=dict(manifest=dict(model=str(path),dtype=args.dtype,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,
        triton=triton.__version__,seed=20260929,attention='forced Flash SDPA',logits_to_keep=1,
        timing='synchronized wall-time, randomized paired rounds; no CUDA Graph',
        baseline='current_best includes earlier small-batch decode_mlp optimization',
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        source_sha256={p:sha(root/p) for p in sources},config_sha256=sha(path/'config.json'),
        weights={p.name:sha(p) for p in sorted(path.glob('*.safetensors'))}),cases=[])
    names=['hf','current_best','qk_fused','compact_gqa','combined']
    target=root/'artifacts'/f'hf_attention_e2e_{args.dtype}{args.tag}.json'
    for batch,seq in [(1,128),(2,128),(1,512)]:
        ids=torch.randint(model.config.vocab_size,(batch,seq+4),device='cuda')
        prompt=ids[:,:seq].contiguous()
        case=dict(batch=batch,seq_len=seq,parity={},timings={})
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            refs={}
            for name in names:
                print('parity',args.dtype,batch,seq,name,flush=True)
                with ctx(name):
                    out=model(prompt,use_cache=True,logits_to_keep=1,output_hidden_states=True)
                    hidden=[t.clone() for t in out.hidden_states]
                    logits=[out.logits.clone()]
                    cache=out.past_key_values
                    for step in range(4):
                        out=model(ids[:,seq+step:seq+step+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
                        cache=out.past_key_values
                        logits.append(out.logits.clone())
                    full=model(ids,use_cache=False,logits_to_keep=1).logits
                    if name in ('hf','current_best'): refs[name]=(hidden,logits)
                    metrics=dict(full_vs_incremental=check(logits[-1],full))
                    for ref_name,(rh,rl) in refs.items():
                        metrics[ref_name]=dict(hidden=[check(a,b) for a,b in zip(hidden,rh)],
                            logits=[check(a,b) for a,b in zip(logits,rl)],
                            argmax=[float((a.argmax(-1)==b.argmax(-1)).float().mean()) for a,b in zip(logits,rl)])
                    metrics['accepted']=metrics['full_vs_incremental']['passes_l2_gate'] and all(
                        p['passes_l2_gate'] for ref_name in refs for group in ('hidden','logits')
                        for p in metrics[ref_name][group])
                    if not metrics['accepted']:
                        print('NUMERICAL GATE FAILED:',name,'(timing diagnostic only)',flush=True)
                    case['parity'][name]=metrics
                    del out,cache,full,hidden,logits
            del refs,rh,rl
            for phase in ('prefill','decode'):
                samples={n:[] for n in names}
                peaks={n:[] for n in names}
                orders=[]
                for repeat in range(args.repeats+4):
                    order=names.copy()
                    rng.shuffle(order)
                    if repeat>=4: orders.append(order)
                    for name in order:
                        with ctx(name):
                            cache=(model(prompt,use_cache=True,logits_to_keep=1).past_key_values
                                   if phase=='decode' else None)
                            tokens=ids[:,seq:seq+1] if phase=='decode' else prompt
                            torch.cuda.synchronize()
                            base=torch.cuda.memory_allocated()
                            torch.cuda.reset_peak_memory_stats()
                            start=time.perf_counter()
                            out=model(tokens,past_key_values=cache,use_cache=True,logits_to_keep=1)
                            torch.cuda.synchronize()
                            ms=(time.perf_counter()-start)*1000
                            if repeat>=4:
                                samples[name].append(ms)
                                peaks[name].append(torch.cuda.max_memory_allocated()-base)
                            del out,cache
                case['timings'][phase]=dict(order=orders,variants={
                    n:dict(median_ms=statistics.median(s),samples_ms=s,peak_delta_bytes=max(peaks[n]),
                           **bootstrap_speedup(samples['current_best'],s)) for n,s in samples.items()})
                print(args.dtype,batch,seq,phase,
                      {n:round(statistics.median(v),3) for n,v in samples.items()},flush=True)
            if args.profile and batch==1 and seq==128:
                case['profiler']={}
                for name in ('current_best','combined'):
                    with ctx(name):
                        from torch.profiler import profile,ProfilerActivity
                        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                            out=model(prompt,use_cache=True,logits_to_keep=1)
                            torch.cuda.synchronize()
                        del out
                    kernels=[e for e in prof.events() if str(e.device_type)=='DeviceType.CUDA']
                    counts=Counter(e.name for e in kernels)
                    times=defaultdict(float)
                    for e in kernels: times[e.name]+=e.device_time_total
                    case['profiler'][name]=dict(cuda_kernel_count=len(kernels),
                        cuda_kernel_sum_us=sum(times.values()),
                        kernels=[dict(name=n,count=c,total_us=times[n]) for n,c in counts.items()],
                        operators=[dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total)
                                   for e in prof.key_averages()])
        report['cases'].append(case)
        target.write_text(json.dumps(report,indent=2)+'\n')
    # Natural-text greedy smoke check, not part of performance measurement.
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    prompt=tokenizer.apply_chat_template([{'role':'user','content':'Explain GPU kernel fusion in one sentence.'}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=tokenizer(prompt,return_tensors='pt').to('cuda')
    generated={}
    for name in ('hf','current_best','combined'):
        with ctx(name),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out=model.generate(**inputs,max_new_tokens=32,do_sample=False,use_cache=True,
                               pad_token_id=tokenizer.eos_token_id)
            generated[name]=out[0,inputs.input_ids.shape[1]:].tolist()
    report['generation_smoke']=dict(prompt=prompt,tokens=generated,
        matches_hf={n:tokens==generated['hf'] for n,tokens in generated.items()},
        text={n:tokenizer.decode(tokens,skip_special_tokens=True) for n,tokens in generated.items()})
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(target,flush=True)


if __name__=='__main__': main()
