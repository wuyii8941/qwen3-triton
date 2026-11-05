import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from ops.triton.add_rms_norm import add_rms_norm_triton


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize('shape',[(1,128),(2,17,2048),(3,129)])
def test_add_norm(dtype,shape):
    torch.manual_seed(9)
    x=torch.randn(*shape,device='cuda',dtype=dtype)
    r=torch.randn_like(x)
    norm=Qwen3RMSNorm(shape[-1]).cuda().to(dtype)
    with torch.inference_mode():
        norm.weight.copy_(torch.randn_like(norm.weight))
        saved=x.clone()
        summed,y=add_rms_norm_triton(x,r,norm.weight,norm.variance_epsilon)
        torch.testing.assert_close(summed,x+r,atol=0,rtol=0)
        torch.testing.assert_close(y,norm(x+r),atol=2e-3,rtol=2e-2)
        torch.testing.assert_close(saved,x,atol=0,rtol=0)
