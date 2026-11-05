import pytest
import torch
from training.hf_rounded import norm,activation

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('width',[128,2048])
def test_rounded_norm_forward_and_backward(dtype,width):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    torch.manual_seed(61)
    module=Qwen3RMSNorm(width,eps=1e-6).cuda().to(dtype)
    with torch.no_grad():module.weight.copy_(torch.randn_like(module.weight))
    x=torch.randn(7,width,device='cuda',dtype=dtype,requires_grad=True)
    upstream=torch.randn_like(x)
    expected=module(x);actual=norm(x,module.weight,1e-6)
    ge=torch.autograd.grad(expected,(x,module.weight),upstream)
    ga=torch.autograd.grad(actual,(x,module.weight),upstream)
    tol=.02 if dtype==torch.bfloat16 else .002
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
    for a,b in zip(ga,ge):torch.testing.assert_close(a,b,atol=0,rtol=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_hf_triton_backward_cast_edges(dtype):
    from training.hf_rounded import activation_triton_backward
    torch.manual_seed(63)
    g=torch.randn(9,513,device='cuda',dtype=dtype,requires_grad=True)
    u=torch.randn_like(g,requires_grad=True)
    grad=torch.randn_like(g)
    expected=torch.nn.functional.silu(g)*u
    actual=activation_triton_backward(g,u)
    ge=torch.autograd.grad(expected,(g,u),grad)
    ga=torch.autograd.grad(actual,(g,u),grad)
    for a,b in zip(ga,ge):
        rel=(a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)
        assert rel < (.005 if dtype==torch.bfloat16 else .001)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_rounded_activation_forward_backward(dtype):
    torch.manual_seed(62)
    g=torch.randn(7,257,device='cuda',dtype=dtype,requires_grad=True)
    u=torch.randn_like(g,requires_grad=True)
    upstream=torch.randn_like(g)
    expected=torch.nn.functional.silu(g)*u;actual=activation(g,u)
    ge=torch.autograd.grad(expected,(g,u),upstream)
    ga=torch.autograd.grad(actual,(g,u),upstream)
    tol=.02 if dtype==torch.bfloat16 else .002
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
    for a,b in zip(ga,ge):torch.testing.assert_close(a,b,atol=0,rtol=0)
