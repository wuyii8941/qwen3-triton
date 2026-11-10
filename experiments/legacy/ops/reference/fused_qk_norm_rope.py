from __future__ import annotations

import torch

from .rms_norm import rms_norm_reference
from .rope import apply_rope_backward_reference, apply_rope_reference


def fused_qk_norm_rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch semantic reference for fused Q/K RMSNorm + RoPE.

    q/k are [batch, seq, heads, head_dim], while cos/sin are
    [batch, seq, head_dim]. The function is differentiable with ordinary
    PyTorch autograd and is intentionally the oracle for a later Triton fuse.
    """
    q_norm = rms_norm_reference(q, q_weight, eps)
    k_norm = rms_norm_reference(k, k_weight, eps)
    q_rotated, k_rotated = apply_rope_reference(
        q_norm.transpose(1, 2),
        k_norm.transpose(1, 2),
        cos,
        sin,
    )
    return q_rotated.transpose(1, 2), k_rotated.transpose(1, 2)


def fused_qk_norm_rope_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit PyTorch backward oracle for fused Q/K RMSNorm + RoPE.

    This function computes the gradients without calling ``autograd.grad``.
    It is intentionally written as readable tensor algebra so a later Triton
    kernel can be checked against each stage of the derivation.
    """
    q_norm = rms_norm_reference(q, q_weight, eps)
    k_norm = rms_norm_reference(k, k_weight, eps)
    grad_q_norm_h, grad_k_norm_h = apply_rope_backward_reference(
        grad_q_out.transpose(1, 2),
        grad_k_out.transpose(1, 2),
        cos,
        sin,
    )
    grad_q_norm = grad_q_norm_h.transpose(1, 2)
    grad_k_norm = grad_k_norm_h.transpose(1, 2)

    def rms_backward(
        x: torch.Tensor,
        weight: torch.Tensor,
        grad_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_f = x.float()
        weight_f = weight.float()
        grad_y_f = grad_y.float()
        inv_rms = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
        weighted_grad = grad_y_f * weight_f
        correction = (weighted_grad * x_f).mean(dim=-1, keepdim=True)
        grad_x = (weighted_grad - x_f * inv_rms.square() * correction) * inv_rms
        grad_weight = (grad_y_f * x_f * inv_rms).sum(dim=tuple(range(x.ndim - 1)))
        return grad_x.to(dtype=x.dtype), grad_weight.to(dtype=weight.dtype)

    grad_q, grad_q_weight = rms_backward(q, q_weight, grad_q_norm)
    grad_k, grad_k_weight = rms_backward(k, k_weight, grad_k_norm)
    return grad_q, grad_k, grad_q_weight, grad_k_weight
