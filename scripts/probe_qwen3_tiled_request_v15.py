"""Research-only S65 tile experiment, leaving production defaults unchanged.

Reuses existing Triton GEMM/gate-up kernels with measured shapes/configurations.
Compare full8B logits with the default route before matched request timings.
"""
import argparse,contextlib,hashlib,json,os,statistics,subprocess,threading,time
from pathlib import Path
from collections import Counter
import torch,triton
from transformers import AutoModelForCausalLM
from qwen3_triton.inference import TritonQwen3Inference,CapturedGreedyRequest,kernels
from qwen3_triton.inference import model as inference_model
from qwen3_triton.kernels.linear import _gemm
from qwen3_triton.kernels.mlp import _gate_up
from qwen3_triton.runtime.memory import raw_empty
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from scripts.compare_qwen3_graph_v15 import hf_greedy,HFCaptured
from scripts.verify_qwen3_inference_v15 import trace

ROUTES=Counter()
@contextlib.contextmanager
def tiled(enabled,model=None):
 if model is not None:
  previous=model.prefill_tiling;model.prefill_tiling=enabled
  try:yield
  finally:model.prefill_tiling=previous
  return
 original_linear=kernels.linear;original_gate=inference_model.gate_up_swiglu
 if not enabled:yield;return
 def linear(x,w,**kw):
  m=x.numel()//x.shape[-1];n,k=w.shape
  if m!=65 or (n,k) not in [(4096,4096),(1024,4096),(4096,12288)]:return original_linear(x,w,**kw)
  bn=32 if n==1024 else 64;ROUTES['linear_'+str((m,n,k))]+=1
  y=raw_empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
  _gemm[(triton.cdiv(m,128)*triton.cdiv(n,bn),)](x,w,y,m,n,k,k,1,1,k,128,bn,64,1,num_warps=4,num_stages=3)
  return y
 def gate(x,wg,wu,**kw):
  m=x.numel()//x.shape[-1];n,k=wg.shape
  if (m,n,k)!=(65,12288,4096):return original_gate(x,wg,wu,**kw)
  ROUTES['gate_up_'+str((m,n,k))]+=1
  shape=(*x.shape[:-1],n);g=raw_empty(shape,device=x.device,dtype=x.dtype);u=raw_empty(shape,device=x.device,dtype=x.dtype);y=raw_empty(shape,device=x.device,dtype=x.dtype)
  _gate_up[(triton.cdiv(m,128)*triton.cdiv(n,32),)](x,wg,wu,g,u,y,m,n,k,128,32,64,1,num_warps=4,num_stages=3,enable_fp_fusion=False)
  return y
 kernels.linear=linear;inference_model.gate_up_swiglu=gate
 try:yield
 finally:kernels.linear=original_linear;inference_model.gate_up_swiglu=original_gate

@torch.inference_mode()
def layout_regressions():
 rows=[]
 for label in ['strided_weight','strided_input']:
  m,n,k=65,4096,4096
  x=torch.randn((m,k*2 if label=='strided_input' else k),device='cuda',dtype=torch.bfloat16)
  w=torch.randn((n,k*2 if label=='strided_weight' else k),device='cuda',dtype=torch.bfloat16)
  if label=='strided_input':x=x[:,::2]
  if label=='strided_weight':w=w[:,::2]
  reference=kernels.linear(x,w,prefill_tiling=False)
  broken=torch.empty_like(reference)
  # Reproduce the pre-review fixed-stride launch, without editing loaded source.
  _gemm[(triton.cdiv(m,128)*triton.cdiv(n,64),)](x,w,broken,m,n,k,k,1,1,k,128,64,64,1,num_warps=4,num_stages=3)
  actual=kernels.linear(x,w,prefill_tiling=True)
  row=dict(case=label,old_fixed_stride_unequal=int((broken!=reference).sum()),guarded_equal_to_default=bool(torch.equal(actual,reference)))
  assert row['old_fixed_stride_unequal']>0 and row['guarded_equal_to_default'],row
  rows.append(row);print(json.dumps(row),flush=True)
 return rows


@torch.inference_mode()
def run(options,out,foreign):
 torch.set_num_threads(2);torch.manual_seed(42)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 root=Path(__file__).resolve().parents[1]
 files=[*sorted((root/'qwen3_triton').rglob('*.py')),Path(__file__),root/'scripts/compare_qwen3_graph_v15.py',root/'scripts/verify_qwen3_inference_v15.py']
 hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 layouts=layout_regressions() if options.production else []
 model_path='/data1/tzh/models/Qwen/Qwen3-8B';c=TritonQwen3Inference.from_pretrained(model_path,prefill_tiling=options.production)
 assert c.prefill_tiling==options.production;c.prefill_tiling=False
 def route(enabled):return tiled(enabled,c if options.production else None)
 checks=[]
 for seed in range(8):
  torch.manual_seed(200+seed);ids=torch.randint(1,100000,(1,69),device='cuda')
  ca=c.new_cache(1,80);cb=c.new_cache(1,80)
  for step in range(5):
   current=ids[:,:65].contiguous() if step==0 else ids[:,64+step:65+step].contiguous()
   a=c(current,past_key_values=ca,logits_to_keep=0).logits
   with route(True):b=c(current,past_key_values=cb,logits_to_keep=0).logits
   row=dict(seed=seed,phase='prefill' if step==0 else 'decode',step=step,bitwise_equal=bool(torch.equal(a,b)),unequal=int((a!=b).sum()),max_abs=float((a.float()-b.float()).abs().max()))
   checks.append(row);print(json.dumps(row),flush=True)
  del ca,cb,a,b
 numerical=all(x['bitwise_equal'] for x in checks)
 ids=torch.randint(1,100000,(1,65),device='cuda');cache=c.new_cache(1,80)
 with route(True):
  _,prefill=trace(out,'prefill',lambda:c(ids,past_key_values=cache),36,warmup_collection=True)
  token=torch.tensor([[50]],device='cuda');_,decode=trace(out,'decode',lambda:c(token,past_key_values=cache),36,warmup_collection=True)
 assert prefill['passed'] and decode['passed'],(prefill,decode)
 h=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
 n=8
 cg=CapturedGreedyRequest(c,ids,max_new_tokens=n)
 with route(True):tg=CapturedGreedyRequest(c,ids,max_new_tokens=n)
 hg=HFCaptured(h,ids,n)
 def eager(x,enable):
  with route(enable):return c.generate(x,max_new_tokens=n,eos_token_id=[],pad_token_id=0)
 funcs={'HF_eager':lambda x:hf_greedy(h,x,n),'HF_graph':hg,'default_eager':lambda x:eager(x,False),'default_graph':cg,'tiled_eager':lambda x:eager(x,True),'tiled_graph':tg}
 comparisons=[]
 for j in range(3):
  x=torch.randint(1,100000,(1,65),device='cuda');ce=eager(x,False);te=eager(x,True);he=hf_greedy(h,x,n)
  row=dict(case=j,default_graph_equal=bool(torch.equal(ce,cg(x))),tiled_graph_equal=bool(torch.equal(te,tg(x))),HF_graph_equal=bool(torch.equal(he,hg(x))),default_vs_tiled_equal=bool(torch.equal(ce,te)),HF_vs_tiled_equal=bool(torch.equal(he,te)))
  comparisons.append(row);assert row['default_graph_equal'] and row['tiled_graph_equal'] and row['HF_graph_equal'],row
 timings=[]
 for order,names in [(1,list(funcs)),(2,list(reversed(funcs)))]:
  for name in names:
   values=[]
   for rep in range(7):
    torch.cuda.synchronize();start=time.perf_counter();funcs[name](ids);torch.cuda.synchronize();elapsed=time.perf_counter()-start
    if rep>=2:values.append(elapsed)
   timings.append(dict(order=order,implementation=name,median_seconds=statistics.median(values),seconds=values))
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(root))],p
 result=dict(complete=True,diagnostic_only=True,production_default_unchanged=True,route_method="model.prefill_tiling" if options.production else "local research override",physical_GPU=1,GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,prompt_tokens=65,output_tokens=8,batch=1,bitwise_equal_to_default=numerical,checks=checks,layout_regressions=layouts,comparisons=comparisons,traces=dict(prefill=prefill,decode=decode),latencies=timings,route_calls=dict(ROUTES),source_sha256=hashes,foreign_context_observations=list(foreign),resource_clean=not foreign,HF_numerical_qualified=False,limitations='Exact M65 8B shapes only; see route_method for API vs research override. B1S65/8tokens/fixedbudget, no earlyEOS. Graph timings exclude preparation. Baseline HF short/decode gates remain failed; no general inference qualification.')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','traces','checks']}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--production',action='store_true');a=p.parse_args();out=a.output.resolve()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and out.is_relative_to(Path('/data1/tzh')) and not out.exists();out.mkdir(parents=True)
 foreign=[];stop=threading.Event()
 with stable_lease([1]):
  uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
  def monitor():
   while not stop.wait(.5):
    try:
     raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
     foreign.extend(line for line in raw.splitlines() if line.split(',')[0].strip()==uuid and int(line.split(',')[1])!=os.getpid())
    except Exception as exc:foreign.append(repr(exc))
  thread=threading.Thread(target=monitor,daemon=True);thread.start()
  try:run(a,out,foreign)
  except BaseException as exc:(out/'failure.json').write_text(json.dumps(dict(error=repr(exc),foreign_context_observations=foreign),indent=2)+'\n');raise
  finally:stop.set();thread.join()
if __name__=='__main__':main()
