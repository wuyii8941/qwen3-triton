import pytest
import torch

from ops.reference.swiglu import swiglu_reference
from ops.triton.silu_and_mul import silu_and_mul_triton


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_silu_and_mul_triton_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(50)
    gate = torch.randn(2, 257, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)
    actual = silu_and_mul_triton(gate, up)
    expected = swiglu_reference(gate, up)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_silu_and_mul_triton_backward_matches_reference() -> None:
    torch.manual_seed(51)
    gate = torch.randn(3, 129, device="cuda", dtype=torch.float32, requires_grad=True)
    up = torch.randn_like(gate, requires_grad=True)
    grad = torch.randn_like(gate)
    actual = silu_and_mul_triton(gate, up)
    actual.backward(grad)
    gate_ref = gate.detach().clone().requires_grad_()
    up_ref = up.detach().clone().requires_grad_()
    swiglu_reference(gate_ref, up_ref).backward(grad)
    torch.testing.assert_close(gate.grad, gate_ref.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(up.grad, up_ref.grad, rtol=1e-5, atol=1e-5)
