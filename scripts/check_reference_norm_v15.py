"""Independent native formula on dense BF16 normalization at short row counts."""
import argparse,json
from pathlib import Path
import torch
from qwen3_triton.inference.kernels import rms_norm
from qwen3_triton.benchmark_utils.stable_idle import stable_lease

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(42);rows=[]
 with stable_lease([1]):
  for count in [1,2,3,5,7,8,15,16,17,65]:
   for sample in range(64):
    x=(torch.randn(count,4096,device='cuda')*torch.exp2(torch.randint(-8,9,(count,4096),device='cuda').float())).to(torch.bfloat16);w=torch.randn(4096,device='cuda',dtype=torch.bfloat16)
    z=x.float();expected=w*(z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)
    actual=rms_norm(x,w,1e-6,reference_accumulation=True)
    if not torch.equal(actual,expected):rows.append(dict(rows=count,sample=sample,unequal=int((actual!=expected).sum()),max_abs=float((actual.float()-expected.float()).abs().max())))
  result=dict(complete=True,cases=640,failures=rows,passed=not rows)
  a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cases=640,failed_cases=len(rows),examples=rows[:4])),flush=True)
  assert not rows,'Short-row RMSNorm differs from independent native formula'
if __name__=='__main__':main()
