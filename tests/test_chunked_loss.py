import pytest
import torch
from torch.nn import functional as F
from training.chunked_loss import chunked_linear_cross_entropy


@pytest.mark.parametrize('chunk',[1,7,64])
@pytest.mark.parametrize('ignored',[False,True])
@pytest.mark.parametrize('backward_impl',['default','native_ce','native_ce_padded'])
def test_full_gradient_reference(chunk,ignored,backward_impl):
    torch.manual_seed(19)
    x=torch.randn(19,16,requires_grad=True)
    w=torch.randn(37,16,requires_grad=True)
    labels=torch.randint(37,(19,))
    if ignored: labels[::3]=-100
    expected=F.cross_entropy(F.linear(x,w).float(),labels)
    actual=chunked_linear_cross_entropy(x,w,labels,chunk_size=chunk,backward_impl=backward_impl)
    ge=torch.autograd.grad(expected,(x,w),torch.tensor(.37))
    ga=torch.autograd.grad(actual,(x,w),torch.tensor(.37))
    torch.testing.assert_close(actual,expected)
    for a,b in zip(ga,ge):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)


def test_tied_embedding_gradient():
    torch.manual_seed(20)
    w=torch.randn(23,8,requires_grad=True)
    ids=torch.tensor([1,2,3,4,5,6])
    x=F.embedding(ids,w)
    expected=F.cross_entropy(F.linear(x,w),ids)
    actual=chunked_linear_cross_entropy(x,w,ids,chunk_size=2)
    a,=torch.autograd.grad(actual,w,retain_graph=True)
    b,=torch.autograd.grad(expected,w)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('backward_impl',['default','native_ce','native_ce_padded'])
def test_partial_gradients_all_ignored_and_invalid_mode(backward_impl):
    for requires_hidden, requires_weight in ((True,False),(False,True),(True,True)):
        x=torch.randn(19,8,requires_grad=requires_hidden)
        w=torch.randn(23,8,requires_grad=requires_weight)
        labels=torch.full((19,),-100,dtype=torch.long)
        loss=chunked_linear_cross_entropy(x,w,labels,chunk_size=7,backward_impl=backward_impl)
        assert torch.isnan(loss)
        loss.backward()
        if requires_hidden:assert torch.count_nonzero(x.grad)==0
        if requires_weight:assert torch.count_nonzero(w.grad)==0
    with pytest.raises(ValueError):
        chunked_linear_cross_entropy(x,w,labels,backward_impl='unknown')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
@pytest.mark.parametrize('vocab',[257,151936])
def test_cuda_full_linear_loss_gradients(dtype,vocab):
    torch.manual_seed(33)
    x=(torch.randn(17,64,device='cuda')*.1).to(dtype).requires_grad_()
    w=(torch.randn(vocab,64,device='cuda')*.1).to(dtype).requires_grad_()
    labels=torch.randint(vocab,(17,),device='cuda');labels[::4]=-100
    expected=F.cross_entropy(F.linear(x,w).float(),labels)
    actual=chunked_linear_cross_entropy(x,w,labels,chunk_size=7)
    ge=torch.autograd.grad(expected,(x,w),torch.tensor(.37,device='cuda'))
    ga=torch.autograd.grad(actual,(x,w),torch.tensor(.37,device='cuda'))
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    for a,b in zip(ga,ge):
        rel=(a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)
        assert rel < (.015 if dtype!=torch.float32 else .0001)
