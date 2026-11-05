import pytest
import torch

from ops.reference.attention import scaled_dot_product_attention_backward_reference
from ops.triton.attention_backward import attention_backward_triton


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_backward_triton_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(20)
    q = torch.randn(2, 4, 17, 32, device="cuda", dtype=dtype)
    k = torch.randn(2, 2, 17, 32, device="cuda", dtype=dtype)
    v = torch.randn(2, 2, 17, 32, device="cuda", dtype=dtype)
    grad_out = torch.randn_like(q)
    expected = scaled_dot_product_attention_backward_reference(
        q, k, v, grad_out, enable_gqa=True
    )
    actual = attention_backward_triton(q, k, v, grad_out)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=5e-2, atol=5e-2)
