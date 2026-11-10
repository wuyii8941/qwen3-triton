import pytest
import torch
from torch.nn import functional as F

from ops.reference.attention import (
    scaled_dot_product_attention_backward_reference,
    scaled_dot_product_attention_reference,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_reference_matches_sdpa(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 4, 7, 16, dtype=dtype)
    k = torch.randn(2, 2, 7, 16, dtype=dtype)
    v = torch.randn(2, 2, 7, 16, dtype=dtype)
    expected = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True
    )
    actual = scaled_dot_product_attention_reference(
        q, k, v, is_causal=True, enable_gqa=True
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_attention_reference_gqa_shape() -> None:
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    v = torch.randn(2, 2, 5, 8)
    output = scaled_dot_product_attention_reference(q, k, v, enable_gqa=True)
    assert output.shape == q.shape


def test_attention_reference_rejects_invalid_gqa() -> None:
    q = torch.randn(1, 3, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    with pytest.raises(ValueError, match="divisible"):
        scaled_dot_product_attention_reference(q, k, v, enable_gqa=True)


def test_attention_explicit_backward_matches_autograd() -> None:
    torch.manual_seed(12)
    q = torch.randn(2, 4, 5, 16, requires_grad=True)
    k = torch.randn(2, 2, 5, 16, requires_grad=True)
    v = torch.randn(2, 2, 5, 16, requires_grad=True)
    grad_out = torch.randn_like(q)
    output = scaled_dot_product_attention_reference(q, k, v, enable_gqa=True)
    torch.autograd.backward(output, grad_out)
    expected = q.grad, k.grad, v.grad
    actual = scaled_dot_product_attention_backward_reference(
        q.detach(), k.detach(), v.detach(), grad_out, enable_gqa=True
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-5)
