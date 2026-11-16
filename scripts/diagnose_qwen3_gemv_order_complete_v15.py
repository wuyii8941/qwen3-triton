"""Research-only real8B/reference-order check with explicit native GEMV trees.

Controlled masks identify sequential32-lane KV and16-lane gate/up accumulators,
followed by descending-offset warp reduction. Loads may unroll without changing
that addition order. The existing fast LMhead path is unchanged.
"""
import argparse,json,os,subprocess,threading
from pathlib import Path
import torch,triton
import triton.language as tl
from qwen3_triton.runtime.memory import raw_empty
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from scripts import diagnose_qwen3_native_order_v15 as base
from scripts.verify_qwen3_inference_v15 import sha
ROOT=Path(__file__).resolve().parents[1]
OLD_ROUTE=base.route;OLD_MATCHED=base.matched_linear

@triton.jit
def _native_gemv_tree(X,W,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,LANES:tl.constexpr,UNROLL:tl.constexpr):
 n=tl.program_id(0)*BN+tl.arange(0,BN);lane=tl.arange(0,LANES)
 acc=tl.zeros((BN,LANES),tl.float32)
 for block in range(tl.cdiv(K,LANES*UNROLL)):
  for sub in tl.static_range(UNROLL):
   k=(block*UNROLL+sub)*LANES+lane
   x=tl.load(X+k,k<K,0).to(tl.float32)
   w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
   acc=tl.fma(w,x[None,:],acc)
 steps:tl.constexpr=5 if LANES==32 else 4
 for step in tl.static_range(steps):
  other=tl.broadcast_to((lane^(1<<(steps-1-step)))[None,:],(BN,LANES))
  acc=acc+tl.gather(acc,other,1)
 value=tl.gather(acc,tl.full((BN,1),0,tl.int32),1).reshape((BN,))
 tl.store(Y+n,value,n<N)

def route(m,n,k):
 if m==1 and k==4096 and n in (1024,12288):return 'gemv_32' if n==1024 else 'gemv_16'
 return OLD_ROUTE(m,n,k)

def matched_linear(x,w,**kw):
 m=x.numel()//x.shape[-1];n,k=w.shape
 if m==1 and k==4096 and n in (1024,12288) and x.is_contiguous() and w.is_contiguous() and not kw.get('accurate',False):
  lanes=32 if n==1024 else 16;bn=4 if lanes==32 else 8
  out=raw_empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
  _native_gemv_tree[(triton.cdiv(n,bn),)](x,w,out,n,k,bn,lanes,8,num_warps=4,enable_fp_fusion=False)
  return out
 if m==1 and k==4096 and n in (1024,12288):return base.ORIGINAL_LINEAR(x,w,**kw)
 return OLD_MATCHED(x,w,**kw)

@torch.inference_mode()
def run(out,foreign):
 files=[*sorted((ROOT/'qwen3_triton').rglob('*.py')),Path(__file__),ROOT/'scripts/diagnose_qwen3_native_order_v15.py',ROOT/'scripts/verify_qwen3_inference_v15.py',ROOT/'scripts/diagnose_qwen3_sliced_coverage_v15.py',ROOT/'scripts/probe_qwen3_native_splitk_v15.py']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
 base.route=route;base.matched_linear=matched_linear
 wrapped=out/'wrapped_check';wrapped.mkdir()
 try:base.run(wrapped,foreign)
 finally:base.route=OLD_ROUTE;base.matched_linear=OLD_MATCHED
 prior=json.loads((wrapped/'result.json').read_text());full=json.loads((wrapped/'original_gates/result.json').read_text())
 for p in files:assert sha(p)==hashes[str(p.relative_to(ROOT))],p
 result=dict(complete=True,diagnostic_only=True,production_changed=False,physical_GPU=1,GPU=torch.cuda.get_device_name(),all_local_calls=len(prior['local_rows']),all_local_matched_unequal=sum(v['matched_unequal'] for v in prior['local_rows']),selected_local_calls=prior['selected_local_calls'],selected_local_matched_unequal=prior['selected_local_matched_unequal'],original_numerical_gates_passed=full['numerics'],greedy_exact_match=full['greedy_exact_match'],wrapped_receipt=dict(path='wrapped_check/result.json',sha256=sha(wrapped/'result.json')),original_verifier_report=dict(path='wrapped_check/original_gates/result.json',sha256=sha(wrapped/'original_gates/result.json')),source_sha256=hashes,foreign_context_observations=list(foreign),resource_clean=not foreign,qualified=False,scope='Reference-only base matched order plus native GEMV32/16 trees with explicit FP32 shuffle order. This outer receipt includes actual new kernel source. Original trace verifier remains unchanged and may reject new kernel names/counts; no API qualification claim.')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
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
  try:run(out,foreign)
  except BaseException as exc:(out/'failure.json').write_text(json.dumps(dict(error=repr(exc),foreign_context_observations=foreign),indent=2)+'\n');raise
  finally:stop.set();thread.join()
if __name__=='__main__':main()
