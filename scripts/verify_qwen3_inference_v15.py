"""Bounded official8B inference audit and opposite-order latency comparison.

SingleGPU/BF16/dense batches. No training artifacts, logits or weights exported.
Retains tiny summaries and at most4x8MiB compressed raw traces per run.
"""
import argparse,gzip,hashlib,json,os,statistics,subprocess,threading,time
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from qwen3_triton.inference import TritonQwen3Inference
from qwen3_triton.benchmark_utils.stable_idle import stable_lease

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(8*2**20),b''):h.update(data)
    return h.hexdigest()

def metrics(x,y):
    x=x.float();y=y.float();d=x-y
    values=dict(max_abs=float(d.abs().max()),mean_abs=float(d.abs().mean()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(y).clamp_min(1e-12)),top1_matches=int((x.argmax(-1)==y.argmax(-1)).sum()),rows=x.numel()//x.shape[-1])
    values['passed']=values['relative_l2']<=.02 and values['mean_abs']<=.03 and values['max_abs']<=.5
    return values


def trace(out,name,fn,expected_cache_appends):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        result=fn();torch.cuda.synchronize()
    p=out/(name+'.json.gz');prof.export_chrome_trace(str(p))
    assert p.stat().st_size<=8*2**20,'Trace exceeds8MiB cap'
    data=json.loads(gzip.decompress(p.read_bytes()));events=data['traceEvents'];counts=Counter(e['name'] for e in events if e.get('cat')=='kernel')
    allowed={'_inference_positions','_inference_cache_append','_inference_cached_attention','_inference_decode_partial','_inference_decode_merge','_inference_gemv','_inference_argmax_parts','_inference_select',
        '_hf_norm_forward','_hf_norm_rope_forward','_hf_swiglu_forward','_gemm','_gate_up','_lookup','_tables','_add',
        '_runtime_fill','_runtime_typed_fill','_runtime_strided_copy','_runtime_binary','_runtime_compare','_runtime_reduce','_runtime_unary'}
    unknown={k:v for k,v in counts.items() if k not in allowed}
    memory=Counter(e.get('name') for e in events if e.get('cat')=='gpu_memset' or (e.get('cat')=='gpu_memcpy' and not e.get('name','').startswith(('Memcpy HtoD','Memcpy DtoH'))))
    report=dict(sha256=sha(p),bytes=p.stat().st_size,kernels=dict(counts),unclassified=unknown,native_GPU_memory_events=dict(memory),expected_cache_appends=expected_cache_appends,
        passed=not unknown and not memory and counts['_inference_cache_append']==expected_cache_appends)
    return result,report


@torch.inference_mode()
def run(a,out,foreign):
    torch.set_num_threads(2);torch.manual_seed(20261116)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    # Verify pinned local config/tokenizer/checkpoint files before loading8B.
    identity=json.loads(Path(a.identity).read_text());model_dir=Path(a.model).resolve()
    assert identity['repo']=='Qwen/Qwen3-8B' and Path(identity['model_dir']).resolve()==model_dir
    for name,item in identity['shards'].items():assert sha(model_dir/name)==item['sha256'],name
    for name,h in identity['model_files_sha256'].items():assert sha(model_dir/name)==h,name
    candidate,setup=trace(out,'setup',lambda:TritonQwen3Inference.from_pretrained(model_dir),0)
    assert sum(p.numel() for p in candidate.parameters())==8190735360 and len(list(candidate.parameters()))==399 and len(candidate.model.layers)==36
    hf=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    cases=[];traces={'setup':setup}
    def save_case(row):
        cases.append(row)
        with (out/'numerical_cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    for batch,length in [(1,17),(1,65),(2,17),(1,257)]:
        ids=torch.randint(1,100000,(batch,length+4),device='cuda');prefix=ids[:,:length].contiguous()
        cache=candidate.new_cache(batch,length+8);hc=DynamicCache()
        x=candidate(prefix,past_key_values=cache,logits_to_keep=0).logits
        y=hf(prefix,past_key_values=hc,use_cache=True,logits_to_keep=0).logits
        values=metrics(x,y);save_case(dict(kind='prefill',batch=batch,length=length,**values))
        assert a.diagnostic or values['passed'],cases[-1]
        for j in range(4):
            token=ids[:,length+j:length+j+1].contiguous()
            x=candidate(token,past_key_values=cache).logits;y=hf(token,past_key_values=hc,use_cache=True,logits_to_keep=1).logits
            values=metrics(x,y);save_case(dict(kind='cached_decode',batch=batch,after_tokens=length+j,**values));assert a.diagnostic or values['passed'],cases[-1]
        # Multi-chunk prefill at a nonzero offset; compare same complete prompt.
        chunks=candidate.new_cache(batch,length+8)
        candidate(prefix[:,:3].contiguous(),past_key_values=chunks)
        x=candidate(prefix[:,3:].contiguous(),past_key_values=chunks,logits_to_keep=1).logits
        y=hf(prefix,use_cache=False,logits_to_keep=1).logits
        values=metrics(x,y);save_case(dict(kind='chunked_prefill',batch=batch,length=length,**values));assert a.diagnostic or values['passed'],cases[-1]
    generation=[]
    for prompt in ['The capital of France is','Write a Python function that adds two numbers.']:
        text=prompt  # Explicit raw-text continuation; local snapshot has no chat template.
        ids=tokenizer(text,return_tensors='pt')['input_ids'].cuda()
        x=candidate.generate(ids,max_new_tokens=16,eos_token_id=[],pad_token_id=0)
        y=hf.generate(ids,max_new_tokens=16,do_sample=False,eos_token_id=None,pad_token_id=0,attention_mask=torch.ones_like(ids),use_cache=True)
        tokens=x[0,ids.shape[1]:].cpu().tolist();reference=y[0,ids.shape[1]:].cpu().tolist()
        row=dict(prompt=prompt,token_ids=tokens,HF_token_ids=reference,exact_match=tokens==reference,text=tokenizer.decode(tokens,skip_special_tokens=True))
        generation.append(row)
    ids=torch.randint(1,100000,(1,65),device='cuda');token=torch.tensor([[50]],device='cuda')
    cache=candidate.new_cache(1,80)
    _,traces['prefill']=trace(out,'prefill',lambda:candidate(ids,past_key_values=cache),36)
    _,traces['decode']=trace(out,'decode',lambda:candidate(token,past_key_values=cache),36)
    _,traces['generate']=trace(out,'generate',lambda:candidate.generate(ids,max_new_tokens=3,eos_token_id=[],pad_token_id=0),108)
    assert a.diagnostic or all(t['passed'] for t in traces.values()),traces
    # Teacher-forced continuation fixed across arms; cache is fresh each sample.
    continuation=torch.randint(1,100000,(1,8),device='cuda');latencies=[]
    for order,names in [(1,['HF','Triton']),(2,['Triton','HF'])]:
        for name in names:
            samples=[]
            for repeat in range(7):
                c=DynamicCache() if name=='HF' else candidate.new_cache(1,80)
                def call(x):return hf(x,past_key_values=c,use_cache=True,logits_to_keep=1) if name=='HF' else candidate(x,past_key_values=c,logits_to_keep=1)
                torch.cuda.synchronize();start=time.perf_counter();call(ids);torch.cuda.synchronize();prefill=time.perf_counter()-start
                steps=[]
                for j in range(8):
                    x=continuation[:,j:j+1].contiguous();torch.cuda.synchronize();start=time.perf_counter();call(x);torch.cuda.synchronize();steps.append(time.perf_counter()-start)
                if repeat>=2:samples.append(dict(prefill_seconds=prefill,decode_seconds=steps))
            latencies.append(dict(order=order,implementation=name,batch=1,prompt_tokens=65,decode_tokens=8,measured_repetitions=5,warmup_repetitions=2,
                prefill_median_seconds=statistics.median(r['prefill_seconds'] for r in samples),decode_median_seconds=statistics.median(t for r in samples for t in r['decode_seconds']),samples=samples))
    sources=[*sorted((ROOT/'qwen3_triton').rglob('*.py')),Path(__file__),ROOT/'scripts/check_qwen3_inference_v15.py',ROOT/'scripts/generate_qwen3_triton_v15.py']
    numerical=all(v['passed'] for v in cases);greedy=all(v['exact_match'] for v in generation)
    result=dict(complete=True,model='Qwen/Qwen3-8B',revision=identity['revision'],parameter_count=8190735360,layers=36,single_GPU=True,
        environment=dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability()),
        physical_GPU=a.physical_device,compute='BF16 weights/activations, FP32 reductions',numerics=numerical,greedy_exact_match=greedy,cases=cases,generation=generation,traces=traces,latencies=latencies,
        strict_GPU_dispatch=candidate.runtime.report(),peak_process_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,foreign_GPU_processes=foreign,
        diagnostic_only=a.diagnostic,qualified=not a.diagnostic and numerical and greedy and all(t['passed'] for t in traces.values()) and not foreign,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},input_identity_sha256=sha(a.identity),
        retained_artifact_policy='Only JSON metrics/token outputs and four raw gzip traces <=8MiB each. No logits/weights/gradients/checkpoints.',
        limitations='Initial singleA6000 dense equal-length BF16 inference: greedy, contiguous request-owned cache, no padding/ragged scheduler, sampling, tensor parallel or serving engine. Finite cases do not establish universal numerical equivalence. Speed is measured, never presumed.')
    (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(complete=True,qualified=result['qualified'],numerics=numerical,greedy_exact_match=greedy,latencies=[{k:v for k,v in r.items() if k!='samples'} for r in latencies])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',default='/data1/tzh/models/Qwen/Qwen3-8B');p.add_argument('--identity',default='/data1/tzh/proj/qwen_operator_lab/artifacts/official_qwen3_inputs_v1.json');p.add_argument('--diagnostic',action='store_true',help='Retain failing metrics and continue; qualification still requires every original gate');p.add_argument('--physical-device',type=int,default=0);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert out.is_relative_to(Path('/data1/tzh')) and not out.exists();out.mkdir(parents=True)
    foreign=[];stop=threading.Event()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.physical_device),'Explicit physical-device/CUDA mapping required'
    with stable_lease([a.physical_device]):
        uuid=subprocess.check_output(['nvidia-smi','-i',str(a.physical_device),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        def monitor():
            while not stop.wait(.5):
                try:
                    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
                    found=[s for s in raw.splitlines() if s.split(',')[0].strip()==uuid and int(s.split(',')[1])!=os.getpid()]
                    if found:foreign.extend(found)
                except Exception as exc:foreign.append(repr(exc))
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try:run(a,out,foreign)
        except BaseException as exc:
            (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),foreign_GPU_processes=foreign),indent=2)+'\n');raise
        finally:stop.set();thread.join()


if __name__=='__main__':main()
