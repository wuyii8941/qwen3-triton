"""Collect complete native linear events and test measured Split-K geometry.

No production change. Include all kernel names, grids, and a bounded raw trace;
previous GEMM-name-only filters cannot establish a complete native schedule.
"""
import argparse,gzip,json,os,subprocess,threading
from pathlib import Path
from collections import Counter
import torch,triton
import triton.language as tl
from qwen3_triton.inference import kernels
from qwen3_triton.kernels.gemm_experiments import split_matmul_into
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from scripts.verify_qwen3_inference_v15 import sha
ROOT=Path(__file__).resolve().parents[1]

@triton.jit
def _ordered_reduce(P,Y,SIZE:tl.constexpr,SPLITS:tl.constexpr,ORDER:tl.constexpr,B:tl.constexpr):
 offset=tl.program_id(0)*B+tl.arange(0,B);value=tl.full((B,),0.,tl.float32)
 for part in tl.static_range(SPLITS):
  index=part if ORDER==0 else SPLITS-1-part
  value=value+tl.load(P+index*SIZE+offset,offset<SIZE,0)
 tl.store(Y+offset,value,offset<SIZE)

def stats(x,y):
 d=x.float()-y.float();return dict(unequal=int((x!=y).sum()),max_abs=float(d.abs().max()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(y.float()).clamp_min(1e-12)))

@torch.inference_mode()
def run(out,foreign):
 torch.set_num_threads(2);torch.manual_seed(20261116)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 files=[*sorted((ROOT/'qwen3_triton').rglob('*.py')),Path(__file__)]
 hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
 shapes=[(m,kind,n,k) for m in [1,17,65,512] for kind,n,k in [('QO',4096,4096),('KV',1024,4096),('gate_up',12288,4096),('down',4096,12288)]]
 scopes=[]
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  with torch.profiler.record_function('bootstrap'):
   for _ in range(64):kernels.positions(1,1,0,torch.device('cuda'))
   torch.cuda.synchronize()
  for m,kind,n,k in shapes:
   x=torch.ones((1,m,k),device='cuda',dtype=torch.bfloat16);w=torch.zeros((n,k),device='cuda',dtype=torch.bfloat16)
   name=f'native_M{m}_{kind}';scopes.append((name,m,kind,n,k))
   torch.nn.functional.linear(x,w);torch.cuda.synchronize()
   with torch.profiler.record_function(name):
    for _ in range(4):torch.nn.functional.linear(x,w)
    torch.cuda.synchronize()
 trace=out/'native_linear.json.gz';prof.export_chrome_trace(str(trace));assert trace.stat().st_size<=8*2**20
 events=json.loads(gzip.decompress(trace.read_bytes()))['traceEvents'];native=[]
 for name,m,kind,n,k in scopes:
  scope=next(e for e in events if e.get('name')==name and e.get('cat')=='user_annotation');start=scope['ts'];end=start+scope['dur']
  launches=[e for e in events if 'LaunchKernel' in e.get('name','') and start<=e.get('ts',-1)<=end]
  correlations={e.get('args',{}).get('correlation') for e in launches}
  gpu=[e for e in events if e.get('cat')=='kernel' and e.get('args',{}).get('correlation') in correlations]
  missing=correlations-{e.get('args',{}).get('correlation') for e in gpu};assert not missing,(name,missing)
  metadata=[]
  for e in gpu:
   item=dict(name=e['name'],grid=e.get('args',{}).get('grid'),block=e.get('args',{}).get('block'),registers=e.get('args',{}).get('registers per thread'),shared_memory=e.get('args',{}).get('shared memory'))
   if item not in metadata:metadata.append(item)
  gemms=[e for e in metadata if any(t in e['name'].lower() for t in ['gemm','gemv','xmma'])]
  assert len(gemms)==1,(name,metadata)
  splits=gemms[0]['grid'][2]
  row=dict(m=m,kind=kind,n=n,k=k,launches=len(launches),GPU_kernels=len(gpu),metadata=metadata,observed_grid_Z=splits)
  native.append(row);print(json.dumps(row),flush=True)
 candidates=[]
 for record in native:
  m,n,k,splits=record['m'],record['n'],record['k'],record['observed_grid_Z']
  if splits<=1:continue
  assert splits<=32,'Unexpected split count; inspect complete metadata'
  x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16);w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16);reference=torch.nn.functional.linear(x,w)
  row=dict(m=m,kind=record['kind'],n=n,k=k,splits=splits,algorithm='current_default',**stats(kernels.linear(x,w),reference));candidates.append(row)
  for grain in [16,32,64,128,256]:
   partial=torch.empty((splits,m,n),device='cuda',dtype=torch.float32);y=torch.empty_like(reference)
   split_matmul_into(x,w.t(),y,partial,(16,32,grain,1,4,2),splits)
   for order in ['tree','forward','reverse']:
    if order!='tree':_ordered_reduce[(triton.cdiv(m*n,256),)](partial,y,m*n,splits,0 if order=='forward' else 1,256,enable_fp_fusion=False)
    row=dict(m=m,kind=record['kind'],n=n,k=k,splits=splits,partition_grain=grain,partition_length=triton.cdiv(k,grain*splits)*grain,algorithm=order,**stats(y,reference))
    candidates.append(row);print(json.dumps(row),flush=True)
 for p in files:assert sha(p)==hashes[str(p.relative_to(ROOT))],p
 result=dict(complete=True,diagnostic_only=True,production_changed=False,physical_GPU=1,GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,native=native,candidates=candidates,raw_trace=dict(file=trace.name,sha256=sha(trace),bytes=trace.stat().st_size),source_sha256=hashes,foreign_context_observations=list(foreign),resource_clean=not foreign,limitations='Native complete-event collection and synthetic random operand probe. gridZ is used as a hypothesis for split count; exact random equality does not qualify complete8B or prove all partition details.')
 raw=json.dumps(result,indent=2)+'\n';assert len(raw.encode())<=2*2**20;(out/'result.json').write_text(raw)
 print(json.dumps(dict(complete=True,split_shapes=len({(v['m'],v['n'],v['k']) for v in candidates}),exact_candidates=sum(v['unequal']==0 for v in candidates),resource_clean=not foreign)),flush=True)

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
