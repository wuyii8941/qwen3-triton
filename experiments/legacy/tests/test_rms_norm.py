import pytest
import torch

from ops.reference.rms_norm import rms_norm_reference
from ops.triton.rms_norm import rms_norm_triton


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="RMSNorm Triton tests require CUDA")


@pytest.mark.parametrize("shape", [(2, 7, 128), (1, 17, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_forward_matches_reference(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    reference = rms_norm_reference(x, weight, 1e-6)
    actual = rms_norm_triton(x, weight, 1e-6)
    torch.testing.assert_close(actual, reference, rtol=3e-3, atol=3e-3)


def test_rms_norm_backward_matches_reference() -> None:
    torch.manual_seed(1)
    x_ref = torch.randn(2, 9, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    weight_ref = torch.randn(128, device="cuda", dtype=torch.float32, requires_grad=True)
    x_custom = x_ref.detach().clone().requires_grad_()
    weight_custom = weight_ref.detach().clone().requires_grad_()
    grad_output = torch.randn_like(x_ref)

    reference = rms_norm_reference(x_ref, weight_ref, 1e-6)
    reference.backward(grad_output)
    actual = rms_norm_triton(x_custom, weight_custom, 1e-6)
    actual.backward(grad_output)

    torch.testing.assert_close(x_custom.grad, x_ref.grad, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(weight_custom.grad, weight_ref.grad, rtol=2e-4, atol=2e-4)
