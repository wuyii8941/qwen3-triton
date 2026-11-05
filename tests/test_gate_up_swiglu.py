import pytest
import torch
from torch.nn import functional as F

from ops.triton.gate_up_swiglu import gate_up_swiglu_triton


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gate_up_swiglu_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(70)
    x = torch.randn(3, 17, 32, device="cuda", dtype=dtype)
    gate_w = torch.randn(64, 32, device="cuda", dtype=dtype)
    up_w = torch.randn_like(gate_w)
    actual = gate_up_swiglu_triton(x, gate_w, up_w)
    expected = F.silu(F.linear(x, gate_w)) * F.linear(x, up_w)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_gate_up_swiglu_backward_matches_reference() -> None:
    torch.manual_seed(71)
    x = torch.randn(2, 9, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    gate_w = torch.randn(64, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    up_w = torch.randn_like(gate_w, requires_grad=True)
    grad = torch.randn(2, 9, 64, device="cuda")
    gate_up_swiglu_triton(x, gate_w, up_w).backward(grad)
    x_ref = x.detach().clone().requires_grad_()
    gw_ref = gate_w.detach().clone().requires_grad_()
    uw_ref = up_w.detach().clone().requires_grad_()
    (F.silu(F.linear(x_ref, gw_ref)) * F.linear(x_ref, uw_ref)).backward(grad)
    for actual, expected in ((x.grad, x_ref.grad), (gate_w.grad, gw_ref.grad), (up_w.grad, uw_ref.grad)):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
