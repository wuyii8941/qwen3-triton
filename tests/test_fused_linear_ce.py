import pytest
import torch
from training.fused_loss import fused_linear_cross_entropy

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('shape',[(17,64,257),(31,128,151936)])
def test_fused_loss_full_gradients(dtype,shape):
    n,h,v=shape
    torch.manual_seed(39)
    x=(torch.randn(n,h,device='cuda')*.1).to(dtype).requires_grad_()
    w=(torch.randn(v,h,device='cuda')*.1).to(dtype).requires_grad_()
    labels=torch.randint(v,(n,),device='cuda');labels[::5]=-100
    expected=torch.nn.functional.cross_entropy(torch.nn.functional.linear(x,w).float(),labels)
    actual=fused_linear_cross_entropy(x,w,labels,chunk_size=7)
    ge=torch.autograd.grad(expected,(x,w))
    ga=torch.autograd.grad(actual,(x,w))
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-4)
    for a,b in zip(ga,ge):
        rel=(a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)
        assert rel < (.015 if dtype!=torch.float32 else .0001)


@pytest.mark.parametrize('which',['hidden','weight','both'])
@pytest.mark.parametrize('ignored',[False,True])
def test_strided_partial_grads_and_all_ignored(which,ignored):
    torch.manual_seed(45)
    x=torch.randn(8,32,device='cuda')[:,::2].detach().requires_grad_(which!='weight')
    w=torch.randn(43,32,device='cuda')[:,::2].detach().requires_grad_(which!='hidden')
    labels=torch.randint(43,(8,),device='cuda')
    if ignored:labels.fill_(-100)
    expected=torch.nn.functional.cross_entropy(torch.nn.functional.linear(x,w),labels)
    actual=fused_linear_cross_entropy(x,w,labels,chunk_size=3)
    torch.testing.assert_close(actual,expected,equal_nan=True)
    inputs=[t for t in (x,w) if t.requires_grad]
    ga=torch.autograd.grad(actual,inputs)
    ge=torch.autograd.grad(expected,inputs)
    for a,b in zip(ga,ge):torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)


def test_fused_tied_weight_gradient():
    torch.manual_seed(46)
    w=torch.randn(37,16,device='cuda',requires_grad=True)
    ids=torch.tensor([1,2,3,4,5,6],device='cuda')
    hidden=torch.nn.functional.embedding(ids,w)
    expected=torch.nn.functional.cross_entropy(torch.nn.functional.linear(hidden,w),ids)
    actual=fused_linear_cross_entropy(hidden,w,ids,chunk_size=2)
    ga,=torch.autograd.grad(actual,w,retain_graph=True)
    ge,=torch.autograd.grad(expected,w)
    torch.testing.assert_close(ga,ge,atol=1e-5,rtol=1e-4)
