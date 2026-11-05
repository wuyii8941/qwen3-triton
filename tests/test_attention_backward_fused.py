import pytest
import torch

from ops.reference.attention import scaled_dot_product_attention_backward_reference
from ops.triton.attention_backward import attention_backward_triton_fused_kv


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention requires CUDA")
def test_attention_backward_fused_kv_matches_reference() -> None:
    torch.manual_seed(21)
    q = torch.randn(2, 4, 17, 32, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 2, 17, 32, device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)
    grad_out = torch.randn_like(q)
    expected = scaled_dot_product_attention_backward_reference(q, k, v, grad_out, enable_gqa=True)
    actual = attention_backward_triton_fused_kv(q, k, v, grad_out)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention requires CUDA")
@pytest.mark.parametrize("seq_len", [17, 64, 129])
def test_register_accumulated_dq_matches_partial_workspace(seq_len: int) -> None:
    """The optimized dQ path must match the workspace baseline.

    Non-multiple lengths are intentional: they exercise the causal boundary
    in the final query/key tiles.
    """
    torch.manual_seed(22 + seq_len)
    q = torch.randn(1, 4, seq_len, 32, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 2, seq_len, 32, device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)
    grad_out = torch.randn_like(q)
    fused = attention_backward_triton_fused_kv(q, k, v, grad_out, fused_dq=True)
    baseline = attention_backward_triton_fused_kv(q, k, v, grad_out, fused_dq=False)
    for fused_tensor, baseline_tensor in zip(fused, baseline):
        torch.testing.assert_close(fused_tensor, baseline_tensor, rtol=2e-4, atol=2e-4)
