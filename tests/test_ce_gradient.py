import pytest
import torch
from ops.triton.ce_gradient import ce_gradient


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
@pytest.mark.parametrize('vocab',[37,151936])
def test_dense_gradient_no_filter(dtype,vocab):
    torch.manual_seed(31)
    x=torch.randn(3,vocab,device='cuda',requires_grad=True)
    target=torch.tensor([0,-100,vocab-1],device='cuda')
    count=(target!=-100).sum()
    up=torch.tensor(.37,device='cuda')
    loss=torch.nn.functional.cross_entropy(x,target)
    expected,=torch.autograd.grad(loss,x,up)
    actual=ce_gradient(x.detach(),x.detach().logsumexp(-1),target,count,up,dtype=dtype)
    torch.testing.assert_close(actual.float(),expected.to(dtype).float(),atol=1e-6,rtol=.01)
