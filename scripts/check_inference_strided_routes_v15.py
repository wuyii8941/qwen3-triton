"""CPU dispatch regression: small positive-stride views must use stride-aware math."""
from contextlib import ExitStack
from unittest.mock import patch
import torch
from qwen3_triton.inference import kernels

class CUDAFixture(torch.Tensor):
 @property
 def is_cuda(self):return True

def main():
 torch.set_num_threads(2);count=0
 for m in [1,8]:
  n=4096 if m==1 else 64;k=4096
  xb=(torch.arange(m*k*2).reshape(m,k*2)%7).to(torch.bfloat16)
  wb=((torch.arange(n*k*2).reshape(n,k*2)%5)-2).to(torch.bfloat16)
  for layout in ['input','weight','both']:
   x=xb[:,::2] if layout in ['input','both'] else xb[:,:k].contiguous()
   w=wb[:,::2] if layout in ['weight','both'] else wb[:,:k].contiguous()
   expected=(x.float()@w.float().t()).to(x.dtype)
   def mm(a,b,**kw):return (a.as_subclass(torch.Tensor).float()@b.as_subclass(torch.Tensor).float()).to(x.dtype)
   def forbidden(*a,**kw):raise AssertionError('A contiguous-only small-row kernel received a noncontiguous view')
   for reference in [False,True]:
    for sliced in [False,True]:
     with ExitStack() as stack:
      gv=stack.enter_context(patch.object(kernels,'_inference_gemv'));gv.__getitem__.side_effect=forbidden
      stack.enter_context(patch.object(kernels,'sliced_linear',side_effect=forbidden))
      stack.enter_context(patch.object(kernels,'raw_empty',side_effect=lambda shape,**kw:torch.empty(shape,dtype=x.dtype)))
      matmul=stack.enter_context(patch.object(kernels,'matmul',side_effect=mm))
      actual=kernels.linear(x.as_subclass(CUDAFixture),w.as_subclass(CUDAFixture),reference_accumulation=reference,sliced=sliced)
      assert matmul.call_count==1 and torch.equal(actual.as_subclass(torch.Tensor),expected),(m,layout,reference,sliced)
     count+=1
 print(f'{count} small-row strided dispatch cases use logical matrix strides for default/reference/sliced options')
if __name__=='__main__':main()
