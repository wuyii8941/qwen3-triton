"""Bounded official8B inference audit and opposite-order latency comparison.

SingleGPU/BF16/dense batches. No training artifacts, logits or weights exported.
Retains tiny summaries and at most5x8MiB compressed raw traces per run.
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

EXECUTION_SOURCES=[*sorted((ROOT/'qwen3_triton').rglob('*.py')),Path(__file__),ROOT/'scripts/check_qwen3_inference_v15.py',ROOT/'scripts/generate_qwen3_triton_v15.py']
EXECUTION_SOURCE_SHA256={str(p.relative_to(ROOT)):sha(p) for p in EXECUTION_SOURCES}

def metrics(x,y):
    x=x.float();y=y.float();d=x-y
    values=dict(max_abs=float(d.abs().max()),mean_abs=float(d.abs().mean()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(y).clamp_min(1e-12)),top1_matches=int((x.argmax(-1)==y.argmax(-1)).sum()),rows=x.numel()//x.shape[-1])
    logp=torch.log_softmax(x,dim=-1);logq=torch.log_softmax(y,dim=-1)
    kl=(logq.exp()*(logq-logp)).sum(-1)
    tv=.5*(logq.exp()-logp.exp()).abs().sum(-1)
    centered=d-d.mean(-1,keepdim=True)
    values.update(KL_HF_to_Triton_mean=float(kl.mean()),KL_max=float(kl.max()),TV_mean=float(tv.mean()),TV_max=float(tv.max()),mean_logit_offset=float(d.mean()),centered_relative_l2=float(torch.linalg.vector_norm(centered)/torch.linalg.vector_norm(y-y.mean(-1,keepdim=True)).clamp_min(1e-12)))
    values['passed']=values['relative_l2']<=.02 and values['mean_abs']<=.03 and values['max_abs']<=.5
    return values


def _phase_counts(kind,*,reference_accumulation=False,sliced_decode=False,prompt_tokens=65):
    families={'_inference_positions','_lookup','_tables','_hf_norm_forward','_hf_norm_rope_forward','_inference_cache_append','_inference_cached_attention','_inference_decode_partial','_inference_decode_merge','_add','_gemm','_gate_up','_inference_gemv','_inference_sliced_linear','_inference_interleaved_linear','_inference_reference_gemv','_split_gemm','_inference_reduce_parts','_hf_swiglu_forward','_inference_argmax_parts','_inference_select'}
    counts=dict.fromkeys(families,0)
    if kind=='setup':return counts
    counts.update(_inference_positions=1,_lookup=1,_tables=1,_hf_norm_forward=73,_hf_norm_rope_forward=72,_inference_cache_append=36,_inference_cached_attention=36,_add=72)
    if kind=='decode':
        counts['_hf_swiglu_forward']=36
        if reference_accumulation:counts.update(_inference_gemv=1,_inference_reference_gemv=144,_inference_interleaved_linear=108)
        elif sliced_decode:counts.update(_inference_gemv=145,_inference_sliced_linear=108)
        else:counts['_inference_gemv']=253
    elif kind=='prefill':
        if prompt_tokens not in (17,65):raise ValueError('Audited B1 prefill length must be17 or65')
        counts['_inference_gemv']=1
        if reference_accumulation and prompt_tokens==17:counts.update(_gemm=108,_inference_interleaved_linear=72,_split_gemm=72,_inference_reduce_parts=72,_hf_swiglu_forward=36)
        elif reference_accumulation:counts.update(_split_gemm=180,_inference_reduce_parts=180,_gate_up=36)
        else:counts.update(_gemm=180,_gate_up=36)
    else:raise ValueError('Explicit measured phase required')
    return counts


def trace(out,name,fn,expected_cache_appends,*,warmup_collection=False,sliced_decode=False,reference_accumulation=False,prompt_tokens=65):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        if warmup_collection:
            from qwen3_triton.inference.kernels import positions
            with torch.profiler.record_function('collection_bootstrap'):
                for _ in range(64):positions(1,1,0,torch.device('cuda'))
                torch.cuda.synchronize()
        with torch.profiler.record_function('actual_'+name):
            result=fn();torch.cuda.synchronize()
    p=out/(name+'.json.gz');prof.export_chrome_trace(str(p))
    assert p.stat().st_size<=8*2**20,'Trace exceeds8MiB cap'
    data=json.loads(gzip.decompress(p.read_bytes()));events=data['traceEvents'];counts=Counter(e['name'] for e in events if e.get('cat')=='kernel')
    allowed={'_inference_positions','_inference_cache_append','_inference_cached_attention','_inference_decode_partial','_inference_decode_merge','_inference_gemv','_inference_sliced_linear','_inference_interleaved_linear','_inference_reference_gemv','_split_gemm','_inference_reduce_parts','_inference_argmax_parts','_inference_select',
        '_hf_norm_forward','_hf_norm_rope_forward','_hf_swiglu_forward','_gemm','_gate_up','_lookup','_tables','_add',
        '_runtime_fill','_runtime_typed_fill','_runtime_strided_copy','_runtime_binary','_runtime_compare','_runtime_reduce','_runtime_unary'}
    scopes=[e for e in events if e.get('name')=='actual_'+name and e.get('cat')=='user_annotation']
    all_counts=dict(counts)
    launches=[e for e in events if 'LaunchKernel' in e.get('name','')]
    if scopes:
        assert len(scopes)==1
        scope=scopes[0];begin=scope['ts'];end=begin+scope['dur']
        launches=[e for e in launches if begin<=e.get('ts',-1)<=end]
        correlations={e.get('args',{}).get('correlation') for e in launches}
        actual=[e for e in events if e.get('cat')=='kernel' and e.get('args',{}).get('correlation') in correlations]
        counts=Counter(e['name'] for e in actual)
    else:actual=[e for e in events if e.get('cat')=='kernel']
    recorded={e.get('args',{}).get('correlation') for e in actual}
    missing=[e.get('args',{}).get('correlation') for e in launches if e.get('args',{}).get('correlation') not in recorded]
    options=dict(reference_accumulation=reference_accumulation,sliced_decode=sliced_decode,prompt_tokens=prompt_tokens)
    if name=='generate':
        assert expected_cache_appends>=36 and expected_cache_appends%36==0
        steps=expected_cache_appends//36
        prefill=_phase_counts('prefill',**options);decode=_phase_counts('decode',**options)
        expected={key:prefill[key]+(steps-1)*decode[key] for key in prefill}
        expected.update(_inference_argmax_parts=steps,_inference_select=steps)
    else:
        if name=='prefill17' and prompt_tokens!=17:raise ValueError('M17 trace must declare17prompt tokens')
        expected=_phase_counts('prefill' if name=='prefill17' else name,**options)
    differences={k:dict(expected=v,actual=counts[k]) for k,v in expected.items() if counts[k]!=v}
    unknown={k:v for k,v in all_counts.items() if k not in allowed}
    memory=Counter(e.get('name') for e in events if e.get('cat')=='gpu_memset' or (e.get('cat')=='gpu_memcpy' and not e.get('name','').startswith(('Memcpy HtoD','Memcpy DtoH'))))
    report=dict(sha256=sha(p),bytes=p.stat().st_size,kernels=dict(counts),all_trace_kernels=all_counts,missing_launch_correlations=missing,count_discrepancies=differences,actual_scope_found=bool(scopes),unclassified=unknown,native_GPU_memory_events=dict(memory),expected_cache_appends=expected_cache_appends,
        passed=bool(scopes) and not unknown and not memory and not missing and not differences and counts['_inference_cache_append']==expected_cache_appends)
    return result,report


def qualification(*,diagnostic,numerics,greedy,traces,foreign,corpus_rows,corpus_blocks,corpus_sequence):
    required={'setup','prefill17','prefill','decode','generate'}
    audits=required.issubset(traces) and all(row['passed'] for row in traces.values())
    functional=not diagnostic and numerics and greedy and audits and not foreign
    corpus_numerics=bool(corpus_rows) and all(row['passed'] for row in corpus_rows)
    full_validation=corpus_blocks==512 and corpus_sequence==512 and len(corpus_rows)==512 and {row['block'] for row in corpus_rows}==set(range(512))
    return dict(functional_qualified=bool(functional),corpus_numerics=corpus_numerics,full_validation=full_validation,qualified=bool(functional and full_validation and corpus_numerics))


@torch.inference_mode()
def run(a,out,foreign):
    torch.set_num_threads(2);torch.manual_seed(20261116)
    reference_accumulation=getattr(a,'reference_accumulation',False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    # Verify pinned local config/tokenizer/checkpoint files before loading8B.
    identity=json.loads(Path(a.identity).read_text());model_dir=Path(a.model).resolve()
    assert identity['repo']=='Qwen/Qwen3-8B' and Path(identity['model_dir']).resolve()==model_dir
    for name,item in identity['shards'].items():assert sha(model_dir/name)==item['sha256'],name
    for name,h in identity['model_files_sha256'].items():assert sha(model_dir/name)==h,name
    candidate,setup=trace(out,'setup',lambda:TritonQwen3Inference.from_pretrained(model_dir,sliced_decode=a.sliced_decode,prefill_tiling=a.prefill_tiling,reference_accumulation=reference_accumulation),0,warmup_collection=a.warmup_collection,sliced_decode=a.sliced_decode,reference_accumulation=reference_accumulation)
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
    corpus_rows=[]
    if a.corpus_blocks:
        from training.data import Blocks
        corpus=Path('/data1/tzh/proj/qwen_operator_lab/artifacts/convergence_corpus_v1/validation.bin')
        assert sha(corpus)==identity['tokens_sha256']['validation.bin']
        blocks=Blocks(corpus,a.corpus_sequence)
        assert 1<=a.corpus_blocks<=len(blocks),'Corpus request must not repeat blocks'
        import math
        assert math.gcd(8191,len(blocks))==1,'Fixed ordering must cover every requested block uniquely'
        for block in range(a.corpus_blocks):
            index=(block*8191)%len(blocks);tokens=blocks[index].unsqueeze(0).cuda()
            x=candidate(tokens,logits_to_keep=0).logits;y=hf(tokens,use_cache=False,logits_to_keep=0).logits
            row=dict(block=index,**metrics(x,y))
            target=tokens[:,1:].reshape(-1)
            row['Triton_NLL']=float(torch.nn.functional.cross_entropy(x[:,:-1].float().reshape(-1,x.shape[-1]),target))
            row['HF_NLL']=float(torch.nn.functional.cross_entropy(y[:,:-1].float().reshape(-1,y.shape[-1]),target))
            corpus_rows.append(row)
            with (out/'corpus_diagnostics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    generation=[]
    for prompt in ['The capital of France is','Write a Python function that adds two numbers.']:
        text=prompt  # Explicit raw-text continuation; local snapshot has no chat template.
        ids=tokenizer(text,return_tensors='pt')['input_ids'].cuda()
        x=candidate.generate(ids,max_new_tokens=16,eos_token_id=[],pad_token_id=0)
        y=hf.generate(ids,max_new_tokens=16,do_sample=False,eos_token_id=None,pad_token_id=0,attention_mask=torch.ones_like(ids),use_cache=True)
        tokens=x[0,ids.shape[1]:].cpu().tolist();reference=y[0,ids.shape[1]:].cpu().tolist()
        row=dict(prompt=prompt,token_ids=tokens,HF_token_ids=reference,exact_match=tokens==reference,text=tokenizer.decode(tokens,skip_special_tokens=True))
        # Teacher-force the same HF continuation, preserving both caches.
        cc=candidate.new_cache(1,ids.shape[1]+16);hc=DynamicCache();current=ids;teacher=[]
        for step,expected in enumerate(reference):
            xc=candidate(current,past_key_values=cc).logits[:,-1,:].float()
            yh=hf(current,past_key_values=hc,use_cache=True,logits_to_keep=1).logits[:,-1,:].float()
            proposed=int(xc.argmax(-1));top=yh.topk(2,dim=-1).values[0]
            teacher.append(dict(step=step,HF=int(yh.argmax(-1)),Triton=proposed,HF_top1_gap=float(top[0]-top[1]),HF_logprob_penalty=float(yh.max()-yh[0,proposed]),**metrics(xc,yh)))
            current=torch.tensor([[expected]],device='cuda')
        row['teacher_forced']=teacher
        generation.append(row)
    ids=torch.randint(1,100000,(1,65),device='cuda');token=torch.tensor([[50]],device='cuda')
    cache=candidate.new_cache(1,80)
    options=dict(warmup_collection=a.warmup_collection,sliced_decode=a.sliced_decode,reference_accumulation=reference_accumulation)
    short_cache=candidate.new_cache(1,32);short_ids=ids[:,:17].contiguous()
    _,traces['prefill17']=trace(out,'prefill17',lambda:candidate(short_ids,past_key_values=short_cache),36,prompt_tokens=17,**options)
    _,traces['prefill']=trace(out,'prefill',lambda:candidate(ids,past_key_values=cache),36,**options)
    _,traces['decode']=trace(out,'decode',lambda:candidate(token,past_key_values=cache),36,**options)
    _,traces['generate']=trace(out,'generate',lambda:candidate.generate(ids,max_new_tokens=3,eos_token_id=[],pad_token_id=0),108,**options)
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
    for source in EXECUTION_SOURCES:assert sha(source)==EXECUTION_SOURCE_SHA256[str(source.relative_to(ROOT))],source
    numerical=all(v['passed'] for v in cases);greedy=all(v['exact_match'] for v in generation)
    acceptance=qualification(diagnostic=a.diagnostic,numerics=numerical,greedy=greedy,traces=traces,foreign=foreign,corpus_rows=corpus_rows,corpus_blocks=a.corpus_blocks,corpus_sequence=a.corpus_sequence)
    result=dict(complete=True,model='Qwen/Qwen3-8B',revision=identity['revision'],parameter_count=8190735360,layers=36,single_GPU=True,
        environment=dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability()),
        physical_GPU=a.physical_device,sliced_decode=a.sliced_decode,prefill_tiling=a.prefill_tiling,reference_accumulation=reference_accumulation,compute='BF16 weights/activations, FP32 reductions',corpus_protocol=dict(sequence=a.corpus_sequence,blocks=a.corpus_blocks,shifted_tokens=a.corpus_blocks*(a.corpus_sequence-1),full_validation=acceptance['full_validation'],unique_blocks=len({row['block'] for row in corpus_rows})),corpus_diagnostics=corpus_rows,numerics=numerical,greedy_exact_match=greedy,cases=cases,generation=generation,traces=traces,latencies=latencies,
        strict_GPU_dispatch=candidate.runtime.report(),peak_process_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,foreign_GPU_processes=foreign,
        diagnostic_only=a.diagnostic,functional_qualified=acceptance['functional_qualified'],corpus_numerics=acceptance['corpus_numerics'],qualified=acceptance['qualified'],
        source_sha256=EXECUTION_SOURCE_SHA256,input_identity_sha256=sha(a.identity),
        retained_artifact_policy='Only JSON metrics/token outputs and five raw gzip traces <=8MiB each. No logits/weights/gradients/checkpoints.',
        limitations='Initial singleA6000 dense equal-length BF16 inference: greedy, contiguous request-owned cache, no padding/ragged scheduler, sampling, tensor parallel or serving engine. Finite cases do not establish universal numerical equivalence. Speed is measured, never presumed.')
    (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(complete=True,qualified=result['qualified'],numerics=numerical,greedy_exact_match=greedy,latencies=[{k:v for k,v in r.items() if k!='samples'} for r in latencies])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',default='/data1/tzh/models/Qwen/Qwen3-8B');p.add_argument('--identity',default='/data1/tzh/proj/qwen_operator_lab/artifacts/official_qwen3_inputs_v1.json');p.add_argument('--sliced-decode',action='store_true');p.add_argument('--prefill-tiling',action='store_true');p.add_argument('--reference-accumulation',action='store_true');p.add_argument('--warmup-collection',action='store_true');p.add_argument('--corpus-sequence',type=int,default=128);p.add_argument('--corpus-blocks',type=int,default=0);p.add_argument('--diagnostic',action='store_true',help='Retain failing metrics and continue; qualification still requires every original gate');p.add_argument('--physical-device',type=int,default=0);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
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
