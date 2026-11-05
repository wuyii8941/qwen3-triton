import pytest
import torch
from torch.nn import functional as F

from ops.reference.attention import scaled_dot_product_attention_reference
from ops.triton.attention import (
    attention_triton,
    attention_triton_autograd,
    attention_triton_decode,
    attention_triton_qwen_prefill,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_triton_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(10)
    q = torch.randn(2, 4, 17, 32, device="cuda", dtype=dtype)
    k = torch.randn(2, 2, 17, 32, device="cuda", dtype=dtype)
    v = torch.randn(2, 2, 17, 32, device="cuda", dtype=dtype)
    expected = scaled_dot_product_attention_reference(q, k, v, enable_gqa=True)
    actual = attention_triton(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention requires CUDA")
def test_attention_triton_matches_sdpa() -> None:
    torch.manual_seed(11)
    q = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    actual = attention_triton(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention requires CUDA")
def test_attention_triton_lse_matches_reference() -> None:
    torch.manual_seed(12)
    q = torch.randn(1, 4, 17, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 17, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2, 17, 32, device="cuda", dtype=torch.bfloat16)
    _, lse = attention_triton(q, k, v, return_lse=True)
    k_expanded = k.repeat_interleave(2, dim=1)
    scores = torch.matmul(q.float(), k_expanded.float().transpose(-2, -1)) / (32 ** 0.5)
    mask = torch.triu(torch.ones(17, 17, device="cuda", dtype=torch.bool), diagonal=1)
    expected = torch.logsumexp(scores.masked_fill(mask, -float("inf")), dim=-1)
    torch.testing.assert_close(lse, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention requires CUDA")
def test_attention_triton_autograd_matches_reference() -> None:
    torch.manual_seed(13)
    q_ref = torch.randn(1, 4, 17, 32, device="cuda", requires_grad=True)
    k_ref = torch.randn(1, 2, 17, 32, device="cuda", requires_grad=True)
    v_ref = torch.randn(1, 2, 17, 32, device="cuda", requires_grad=True)
    q_custom = q_ref.detach().clone().requires_grad_()
    k_custom = k_ref.detach().clone().requires_grad_()
    v_custom = v_ref.detach().clone().requires_grad_()
    grad_out = torch.randn_like(q_ref)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=True, enable_gqa=True
    )
    expected.backward(grad_out)
    actual = attention_triton_autograd(q_custom, k_custom, v_custom)
    actual.backward(grad_out)
    torch.testing.assert_close(q_custom.grad, q_ref.grad, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(k_custom.grad, k_ref.grad, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(v_custom.grad, v_ref.grad, rtol=8e-2, atol=8e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention requires CUDA")
def test_qwen_prefill_keeps_gqa_compact() -> None:
    torch.manual_seed(14)
    q = torch.randn(1, 16, 129, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 129, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    actual = attention_triton_qwen_prefill(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    torch.testing.assert_close(actual, expected, rtol=6e-2, atol=6e-2)
    assert k.shape[1] == 8 and v.shape[1] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention requires CUDA")
def test_decode_kernel_matches_compact_gqa_reference() -> None:
    torch.manual_seed(15)
    q = torch.randn(2, 16, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 8, 257, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    actual = attention_triton_decode(q, k, v)
    k_expanded = k.repeat_interleave(2, dim=1)
    scores = torch.matmul(q.float(), k_expanded.float().transpose(-2, -1)) / (128 ** 0.5)
    expected = torch.softmax(scores, dim=-1) @ v.repeat_interleave(2, dim=1).float()
    torch.testing.assert_close(actual.float(), expected, rtol=6e-2, atol=6e-2)
