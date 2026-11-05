"""Compare fullgraph Inductor with explicit Graph on the same cache/backend.

Both prefill and decode are compiled; Dynamo backend invocations are counted.
No silent graph-break fallback, no double CUDAGraph capture, no timing a first
compile. The Torch reference keeps HF operators but adapts static cache/GQA.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
import torch
import triton
import transformers
from torch._dynamo.backends.registry import lookup_backend
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoModelForCausalLM
from model.hf_inference import qwen3_triton_inference,pack_qkv_weights
from model.static_decode import StaticDecodeRunner
from model.torch_static_reference import qwen3_torch_static_reference
from model.hybrid_decode import HybridDecodeRunner
from benchmarks.benchmark_hf_attention_e2e import check,bootstrap_speedup


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',default='1:128')
    p.add_argument('--repeats',type=int,default=12)
    p.add_argument('--tag',default='')
    p.add_argument('--hybrid',action='store_true')
    args=p.parse_args()
    if args.repeats<8: raise ValueError('At least eight rounds')
    target=Path(f'artifacts/compile_pipeline{args.tag}.json')
    if target.exists():raise FileExistsError(target)
    torch.manual_seed(20261006)
    model_path=Path('/data1/tzh/models/Qwen/Qwen3-1.7B')
    model=AutoModelForCausalLM.from_pretrained(model_path,
        local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    def ctx(name, instance):
        if name.startswith('torch_'): return qwen3_torch_static_reference(instance)
        return qwen3_triton_inference(instance,packed_weights=pack_qkv_weights(instance),decode_mlp=True,
            fuse_qk=True,compact_gqa=True,static_decode=True,packed_qkv=True,
            direct_k_cache=True,direct_v_cache=True,prefill_mlp=True,fuse_residual=True)
    names=['torch_graph','torch_compiled_graph','triton_graph','triton_compiled_graph']
    if args.hybrid: names.append('hybrid_graph')
    options={'triton.cudagraphs':False,'compile_threads':4,'emulate_precision_casts':True}
    source=['model/torch_static_reference.py','model/hf_inference.py','model/static_decode.py',
            'model/hf_attention.py','ops/triton/hf_qk_norm_rope.py','ops/compile_ops.py',
            'ops/triton/gemm_tuning.py','ops/triton/decode_gate_up.py',
            'ops/triton/add_rms_norm.py','model/hybrid_decode.py',__file__]
    report=dict(manifest=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,triton=triton.__version__,transformers=transformers.__version__,
        capability=torch.cuda.get_device_capability(),dtype='float16',model=str(model_path),
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        config_sha256=hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest(),
        repeats=args.repeats,decode_steps=16,setup_scope='first forwards, compilation/cache lookup, warmup and CUDA Graph capture; model load/packing excluded',
        seed=20261006,fullgraph=True,dynamic=False,inductor_options=options,
        attention='native Flash, compact GQA, identical static-cache contract',
        torch_reference='HF operators; adapted static-cache/attention plumbing, NOT untouched HF generate',
        timing='prefill + 16 greedy decode steps; explicit CUDA Graph decode in ALL variants',
        compile_boundary='explicit inference custom ops preserve no-FMA kernels and cache mutations; NOT compiler-fused across opaque ops',
        model_ownership='separate stable module trees sharing read-only original Parameter tensors; contexts stay installed throughout timing',
        hybrid='compiled pure-Torch prefill + eager Triton decode captured as CUDA Graph, shared KV addresses' if args.hybrid else None,
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in source}),cases=[])
    rng=random.Random(20261006)
    for batch,seq in (tuple(map(int,c.split(':'))) for c in args.cases.split(',')):
        ids=torch.randint(model.config.vocab_size,(batch,seq+8),device='cuda')
        prompt=ids[:,:seq].contiguous()
        case=dict(batch=batch,seq=seq,setup={},parity={},timings={})
        runners={}
        counters={n:[] for n in names}
        models={}
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION), ExitStack() as contexts:
            # Untouched HF is the numerical oracle, not a timed variant here.
            out=model(prompt,use_cache=True,logits_to_keep=1)
            cache=out.past_key_values
            reference=[out.logits.clone()]
            for i in range(8):
                out=model(ids[:,seq+i:seq+i+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
                reference.append(out.logits.clone())
            del out,cache
            for name in names:
                print('setup',batch,seq,name,flush=True)
                # Rebinding a context every round invalidates Dynamo guards.
                # Keep separate module objects, but share immutable weights.
                if name=='hybrid_graph':
                    runner=HybridDecodeRunner(models['torch_graph'],models['triton_graph'],
                                              batch,2**(seq+16-1).bit_length())
                else:
                    instance=copy.deepcopy(model,{id(p):p for p in model.parameters()})
                    models[name]=instance
                    contexts.enter_context(ctx(name,instance))
                    runner=StaticDecodeRunner(instance,batch,2**(seq+16-1).bit_length())
                if 'compiled' in name or name=='hybrid_graph':
                    def backend(gm,inputs,name=name):
                        counters[name].append(dict(nodes=len(list(gm.graph.nodes))))
                        return lookup_backend('inductor')(gm,inputs,config_patches=options)
                    if name=='hybrid_graph': runner.compile_prefill(backend=backend)
                    else: runner._forward=torch.compile(runner._forward,backend=backend,fullgraph=True,dynamic=False)
                torch.cuda.synchronize()
                start=time.perf_counter()
                try:
                    runner.prefill(prompt)
                    runner.capture(ids[:,seq:seq+1])
                    torch.cuda.synchronize()
                    case['setup'][name]=dict(ms=(time.perf_counter()-start)*1000,success=True)
                    logits=[runner.prefill(prompt).clone()]
                    for i in range(8): logits.append(runner.decode(ids[:,seq+i:seq+i+1]).clone())
                    metrics=[check(a,b) for a,b in zip(logits,reference)]
                    case['parity'][name]=dict(logits=metrics,accepted=all(v['passes_l2_gate'] for v in metrics),
                        argmax_match=[bool(torch.equal(a.argmax(-1),b.argmax(-1))) for a,b in zip(logits,reference)])
                    runners[name]=runner
                    print('parity',name,case['parity'][name]['accepted'],max(v['relative_l2'] for v in metrics),flush=True)
                except Exception as exc:
                    case['setup'][name]=dict(ms=(time.perf_counter()-start)*1000,success=False,
                        error_type=type(exc).__name__,error=str(exc))
                    print('FAILED',name,type(exc).__name__,str(exc)[:1600],flush=True)
                # Preserve partial setup results if any later experiment fails.
                target.write_text(json.dumps(dict(report,current_case=case),indent=2)+'\n')
            live=list(runners)
            counts_before={n:len(counters[n]) for n in live}
            samples={phase:{n:[] for n in live} for phase in ('prefill','decode','pipeline')}
            for repeat in range(args.repeats+3):
                order=live.copy();rng.shuffle(order)
                for name in order:
                    r=runners[name]
                    torch.cuda.synchronize();start=time.perf_counter()
                    logits=r.prefill(prompt)
                    torch.cuda.synchronize();pre=(time.perf_counter()-start)*1000
                    dec_start=time.perf_counter()
                    for _ in range(16): logits=r.decode(logits[:,-1].argmax(-1,keepdim=True))
                    torch.cuda.synchronize()
                    dec=(time.perf_counter()-dec_start)*1000
                    total=(time.perf_counter()-start)*1000
                    if repeat>=3:
                        for phase,t in [('prefill',pre),('decode',dec),('pipeline',total)]:samples[phase][name].append(t)
                if repeat>=3 and (repeat-2)%4==0:print('rounds',batch,seq,repeat-2,flush=True)
            case['compile_graphs']=counters
            case['compile_counts_before_timing']=counts_before
            case['no_recompile_during_timing']={n:len(counters[n])==counts_before[n] for n in live}
            case['valid_timings']={n:case['parity'][n]['accepted'] and
                case['no_recompile_during_timing'][n] for n in live}
            for phase,values in samples.items():
                case['timings'][phase]={n:dict(median_ms=statistics.median(v),samples_ms=v,
                    **bootstrap_speedup(values['torch_graph'],v)) for n,v in values.items()}
                for n,v in values.items():
                    if 'triton_graph' in values:
                        case['timings'][phase][n]['vs_triton_graph']=bootstrap_speedup(values['triton_graph'],v)
                    if 'torch_compiled_graph' in values:
                        case['timings'][phase][n]['vs_torch_compiled_graph']=bootstrap_speedup(values['torch_compiled_graph'],v)
            # Profiler after timing only; also confirm which kernels were run.
            from torch.profiler import profile,ProfilerActivity
            case['profiler']={}
            for name in live:
                r=runners[name];r.prefill(prompt);torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                    r.decode(ids[:,seq:seq+1]);torch.cuda.synchronize()
                events=[e for e in prof.events() if str(e.device_type)=='DeviceType.CUDA']
                case['profiler'][name]=dict(kernel_count=len(events),
                    kernels=dict(Counter(e.name for e in events)),sum_us=sum(e.device_time_total for e in events))
            report['cases'].append(case)
            target.write_text(json.dumps(report,indent=2)+'\n')
            for phase,rows in case['timings'].items():print(phase,{n:round(v['median_ms'],3) for n,v in rows.items()},flush=True)
            print('compile counts',counts_before,case['no_recompile_during_timing'],flush=True)
        del runners,runner,r,models,instance,contexts
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    print(target,flush=True)


if __name__=='__main__':main()
