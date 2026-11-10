import pytest
import torch

from ops.reference.fused_qk_norm_rope import (
    fused_qk_norm_rope_backward_reference,
    fused_qk_norm_rope_reference,
)
from ops.reference.rms_norm import rms_norm_reference
from ops.reference.rope import apply_rope_reference
from ops.triton.fused_qk_norm_rope import fused_qk_norm_rope_triton


def _independent_path(q, k, q_weight, k_weight, cos, sin, eps):
    q_norm = rms_norm_reference(q, q_weight, eps)
    k_norm = rms_norm_reference(k, k_weight, eps)
    q_rotated, k_rotated = apply_rope_reference(
        q_norm.transpose(1, 2), k_norm.transpose(1, 2), cos, sin
    )
    return q_rotated.transpose(1, 2), k_rotated.transpose(1, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_reference_matches_independent_path(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 7, 4, 16, dtype=dtype)
    k = torch.randn(2, 7, 2, 16, dtype=dtype)
    q_weight = torch.randn(16, dtype=dtype)
    k_weight = torch.randn(16, dtype=dtype)
    cos = torch.randn(2, 7, 16, dtype=dtype)
    sin = torch.randn(2, 7, 16, dtype=dtype)
    expected = _independent_path(q, k, q_weight, k_weight, cos, sin, 1e-6)
    actual = fused_qk_norm_rope_reference(q, k, q_weight, k_weight, cos, sin, 1e-6)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_fused_reference_backward() -> None:
    torch.manual_seed(1)
    q = torch.randn(2, 5, 4, 16, requires_grad=True)
    k = torch.randn(2, 5, 2, 16, requires_grad=True)
    q_weight = torch.randn(16, requires_grad=True)
    k_weight = torch.randn(16, requires_grad=True)
    cos = torch.randn(2, 5, 16)
    sin = torch.randn(2, 5, 16)
    q_out, k_out = fused_qk_norm_rope_reference(q, k, q_weight, k_weight, cos, sin)
    loss = q_out.square().mean() + k_out.square().mean()
    loss.backward()
    for tensor in (q.grad, k.grad, q_weight.grad, k_weight.grad):
        assert tensor is not None
        assert torch.isfinite(tensor).all()


def test_fused_explicit_backward_matches_autograd() -> None:
    torch.manual_seed(11)
    q = torch.randn(2, 5, 4, 16, requires_grad=True)
    k = torch.randn(2, 5, 2, 16, requires_grad=True)
    q_weight = torch.randn(16, requires_grad=True)
    k_weight = torch.randn(16, requires_grad=True)
    cos = torch.randn(2, 5, 16)
    sin = torch.randn(2, 5, 16)
    grad_q = torch.randn_like(q)
    grad_k = torch.randn_like(k)

    out_q, out_k = fused_qk_norm_rope_reference(
        q, k, q_weight, k_weight, cos, sin
    )
    torch.autograd.backward((out_q, out_k), (grad_q, grad_k))
    expected = (q.grad, k.grad, q_weight.grad, k_weight.grad)
    actual = fused_qk_norm_rope_backward_reference(
        q.detach(),
        k.detach(),
        q_weight.detach(),
        k_weight.detach(),
        cos,
        sin,
        grad_q,
        grad_k,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused Triton test requires CUDA")
def test_fused_triton_matches_reference() -> None:
    torch.manual_seed(2)
    dtype = torch.bfloat16
    q = torch.randn(2, 7, 4, 16, device="cuda", dtype=dtype)
    k = torch.randn(2, 7, 2, 16, device="cuda", dtype=dtype)
    q_weight = torch.randn(16, device="cuda", dtype=dtype)
    k_weight = torch.randn(16, device="cuda", dtype=dtype)
    cos = torch.randn(2, 7, 16, device="cuda", dtype=dtype)
    sin = torch.randn(2, 7, 16, device="cuda", dtype=dtype)
    expected = fused_qk_norm_rope_reference(q, k, q_weight, k_weight, cos, sin)
    actual = fused_qk_norm_rope_triton(q, k, q_weight, k_weight, cos, sin)
    torch.testing.assert_close(actual[0], expected[0], rtol=3e-2, atol=3.2e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=3e-2, atol=3.2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused Triton test requires CUDA")
def test_fused_triton_backward_matches_reference() -> None:
    torch.manual_seed(3)
    q_ref = torch.randn(2, 5, 4, 16, device="cuda", requires_grad=True)
    k_ref = torch.randn(2, 5, 2, 16, device="cuda", requires_grad=True)
    q_weight_ref = torch.randn(16, device="cuda", requires_grad=True)
    k_weight_ref = torch.randn(16, device="cuda", requires_grad=True)
    q_custom = q_ref.detach().clone().requires_grad_()
    k_custom = k_ref.detach().clone().requires_grad_()
    q_weight_custom = q_weight_ref.detach().clone().requires_grad_()
    k_weight_custom = k_weight_ref.detach().clone().requires_grad_()
    cos = torch.randn(2, 5, 16, device="cuda")
    sin = torch.randn(2, 5, 16, device="cuda")
    grad_q = torch.randn_like(q_ref)
    grad_k = torch.randn_like(k_ref)

    ref_q, ref_k = fused_qk_norm_rope_reference(
        q_ref, k_ref, q_weight_ref, k_weight_ref, cos, sin
    )
    torch.autograd.backward((ref_q, ref_k), (grad_q, grad_k))
    custom_q, custom_k = fused_qk_norm_rope_triton(
        q_custom, k_custom, q_weight_custom, k_weight_custom, cos, sin
    )
    torch.autograd.backward((custom_q, custom_k), (grad_q, grad_k))

    for actual, expected in (
        (q_custom.grad, q_ref.grad),
        (k_custom.grad, k_ref.grad),
        (q_weight_custom.grad, q_weight_ref.grad),
        (k_weight_custom.grad, k_weight_ref.grad),
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
