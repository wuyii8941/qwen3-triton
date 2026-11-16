"""Functional dunder boolean dispatch must not select an in-place output."""
import torch
from unittest.mock import patch
from qwen3_triton.runtime.dispatch import TritonRuntimeMode
class CUDAFixture(torch.Tensor):
 @property
 def is_cuda(self):return True

def main():
 a=torch.tensor([True,False]).as_subclass(CUDAFixture);b=torch.tensor([False,True]).as_subclass(CUDAFixture)
 mode=TritonRuntimeMode()
 for op in [torch.ops.aten.__and__.Tensor,torch.ops.aten.__or__.Tensor]:
  with patch('qwen3_triton.runtime.dispatch.ops.binary',return_value=None) as call:
   mode.__torch_dispatch__(op,[],(a,b))
   assert call.call_args.kwargs['out'] is None,f'{op} incorrectly aliases/mutates its left operand'
 with patch('qwen3_triton.runtime.dispatch.ops.binary',return_value=None) as call:
  mode.__torch_dispatch__(torch.ops.aten.add_.Tensor,[],(a,b))
  assert call.call_args.kwargs['out'] is a,'Real in-place operation lost its output'
 print('Functional boolean dunders do not alias inputs; real in-place dispatch retained')
if __name__=='__main__':main()
