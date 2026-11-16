"""Bounded inference tile experiment; production dispatch is unchanged.

Hypothesis: at M65, M128 tiles avoid rereading weights across two M64 tiles.
Every candidate is compared bitwise with the current kernel on identical operands.
Graph replay timings exclude compilation, allocation, and capture.
"""
import argparse,json,hashlib,os,subprocess,threading
from pathlib import Path
import torch,triton
from qwen3_triton.kernels.linear import _gemm
from qwen3_triton.kernels.mlp import _gate_up
from qwen3_triton.benchmark_utils.stable_idle import stable_lease

@torch.inference_mode()
def run(a,foreign):
 torch.set_num_threads(2);torch.manual_seed(20261116)
 configs=[(64,64,32,8,4,3),(128,32,32,1,4,3),(128,64,32,1,4,3),(64,32,32,1,4,3),(32,64,32,1,4,3),(128,32,32,1,4,2),(128,32,64,1,4,3),(128,64,64,1,4,3)]
 rows=[]
 for m in [65,17,257]:
  for kind,n,k in [('QO',4096,4096),('KV',1024,4096),('down',4096,12288),('gate_up',12288,4096)]:
   x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16);w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16);u=torch.randn_like(w) if kind=='gate_up' else None
   y=torch.empty((m,n),device='cuda',dtype=torch.bfloat16);ref=torch.empty_like(y)
   g=torch.empty_like(y);up=torch.empty_like(y)
   def launch(cfg,out):
    bm,bn,bk,group,warps,stages=cfg;grid=(triton.cdiv(m,bm)*triton.cdiv(n,bn),)
    if kind=='gate_up':_gate_up[grid](x,w,u,g,up,out,m,n,k,bm,bn,bk,group,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    else:_gemm[grid](x,w,out,m,n,k,k,1,1,k,bm,bn,bk,group,num_warps=warps,num_stages=stages)
   launch(configs[0],ref)
   for cfg in configs:
    launch(cfg,y);torch.cuda.synchronize()
    unequal=int((ref!=y).sum());d=(y.float()-ref.float()).abs()
    ms=triton.testing.do_bench_cudagraph(lambda:launch(cfg,y),rep=100)
    row=dict(m=m,n=n,k=k,kind=kind,config=cfg,unequal_vs_default=unequal,max_abs=float(d.max()),mean_abs=float(d.mean()),graph_ms=ms)
    rows.append(row);print(json.dumps(row),flush=True)
 result=dict(complete=True,diagnostic_only=True,physical_GPU=1,torch=torch.__version__,triton=triton.__version__,GPU=torch.cuda.get_device_name(),rows=rows,foreign_context_observations=list(foreign),resource_clean=not foreign,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('qwen3_triton/kernels/linear.py'),Path('qwen3_triton/kernels/mlp.py')]},limitations='Synthetic identical BF16 inputs, one layer shape at a time, no whole-model numerical or speed qualification.')
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and out.is_relative_to(Path('/data1/tzh')) and not out.exists()
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
  try:run(a,foreign)
  finally:stop.set();thread.join()
if __name__=='__main__':main()
