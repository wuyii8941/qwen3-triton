import pytest
import torch
from transformers.loss.loss_utils import ForCausalLMLoss
from benchmarks.chunked_ce_only_probe import chunked_causal_ce,gradient_difference


@pytest.mark.parametrize('chunk',[1,3,128])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
def test_native_head_rows_and_ce_gradients(chunk,dtype):
    torch.manual_seed(8)
    x=torch.randn(2,7,11,dtype=dtype,requires_grad=True)
    labels=torch.randint(11,(2,7));labels[0,2]=-100
    ref=ForCausalLMLoss(x,labels,11);actual=chunked_causal_ce(x,labels,11,chunk_size=chunk)
    torch.testing.assert_close(ref,actual)
    upstream=torch.tensor(.37)
    a,=torch.autograd.grad(ref,x,upstream,retain_graph=True)
    b,=torch.autograd.grad(actual,x,upstream)
    torch.testing.assert_close(a,b)


def test_all_ignored_and_reject_unsupported():
    x=torch.randn(2,3,7,requires_grad=True);labels=torch.full((2,3),-100)
    loss=chunked_causal_ce(x,labels,7,chunk_size=2)
    assert torch.isnan(loss)
    loss.backward();assert torch.count_nonzero(x.grad)==0
    for options in (dict(chunk_size=0),dict(chunk_size=True),dict(num_items_in_batch=4),dict(unknown=True)):
        with pytest.raises(ValueError):chunked_causal_ce(x,labels,7,**options)


def test_bounded_gradient_diagnostic():
    x=torch.tensor([0.,1.,2.],dtype=torch.bfloat16)
    assert gradient_difference(x,x)==dict(max_abs=0.,failed_elements=0)
    y=x.clone();y[0]=1
    assert gradient_difference(x,y)==dict(max_abs=1.,failed_elements=1)
    with pytest.raises(ValueError):gradient_difference(x,x[:1])
