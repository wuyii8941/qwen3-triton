from __future__ import annotations

import torch
import triton
import triton.language as tl

from ops.reference.fused_qk_norm_rope import fused_qk_norm_rope_reference


@triton.jit
def _fused_qk_norm_rope_kernel(
    x_ptr,
    y_ptr,
    cos_ptr,
    sin_ptr,
    w_ptr,
    n_heads,
    seq_len,
    head_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BSHD: tl.constexpr = False,
):
    """Fuse RMSNorm and Qwen3 split-half RoPE for one Q or K tensor.

    x/y are laid out as [batch, heads, seq, head_dim]. cos/sin are laid out
    as [batch, seq, head_dim]. One program handles one batch/head/sequence
    position and keeps the normalized values in registers before rotating.
    """
    pid = tl.program_id(0)
    tokens_per_batch = n_heads * seq_len
    batch = pid // tokens_per_batch
    rem = pid % tokens_per_batch
    head = rem // seq_len
    seq = rem % seq_len

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim
    half = head_dim // 2
    paired = tl.where(offsets < half, offsets + half, offsets - half)
    sign = tl.where(offsets < half, -1.0, 1.0)

    x_base = ((batch * n_heads + head) * seq_len + seq) * head_dim
    if BSHD:
        # Native projection layout. Adjacent programs now process adjacent
        # heads; neither input nor output needs a transpose-copy.
        batch = pid // (seq_len * n_heads)
        seq = (pid // n_heads) % seq_len
        x_base = pid * head_dim
    pos_base = (batch * seq_len + seq) * head_dim

    x_value = tl.load(x_ptr + x_base + offsets, mask=mask, other=0.0).to(tl.float32)
    paired_x_value = tl.load(x_ptr + x_base + paired, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    paired_weight = tl.load(w_ptr + paired, mask=mask, other=0.0).to(tl.float32)

    mean_square = tl.sum(x_value * x_value, axis=0) / head_dim
    inv_rms = tl.rsqrt(mean_square + eps)
    normalized = x_value * inv_rms * weight
    paired_normalized = paired_x_value * inv_rms * paired_weight

    cos_value = tl.load(cos_ptr + pos_base + offsets, mask=mask, other=0.0).to(tl.float32)
    sin_value = tl.load(sin_ptr + pos_base + offsets, mask=mask, other=0.0).to(tl.float32)
    rotated = normalized * cos_value + (paired_normalized * sign) * sin_value
    tl.store(y_ptr + x_base + offsets, rotated, mask=mask)


def _fused_one(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, weight, cos, sin)):
        raise ValueError("fused_norm_rope_triton requires CUDA tensors")
    if x.ndim != 4 or cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("expected x=[batch,seq,heads,head_dim], cos/sin=[batch,seq,head_dim]")
    batch, seq_len, heads, head_dim = x.shape
    if cos.shape != (batch, seq_len, head_dim):
        raise ValueError("cos/sin shape must be [batch,seq,head_dim]")
    if weight.shape != (head_dim,):
        raise ValueError("weight shape must be [head_dim]")
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")

    # Projection outputs already have contiguous [B,S,H,D] layout.
    x_kernel = x.contiguous()
    y_kernel = torch.empty_like(x_kernel)
    cos = cos.contiguous()
    sin = sin.contiguous()
    block_size = triton.next_power_of_2(head_dim)
    if block_size > 65536:
        raise ValueError("head_dim is too large for this first kernel")
    _fused_qk_norm_rope_kernel[(batch * heads * seq_len,)](
        x_kernel,
        y_kernel,
        cos,
        sin,
        weight.contiguous(),
        heads,
        seq_len,
        head_dim,
        eps,
        BLOCK_SIZE=block_size,
        BSHD=True,
    )
    return y_kernel

@triton.jit
def _rope_backward_kernel(
    x_ptr,
    grad_y_ptr,
    cos_ptr,
    sin_ptr,
    w_ptr,
    grad_x_ptr,
    grad_w_ptr,
    n_heads,
    seq_len,
    head_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward kernel for split-half RoPE.

    For ``y = x*cos + rotate_half(x)*sin``, this computes
    ``grad_x = grad_y*cos + rotate_half_transpose(grad_y*sin)``.
    The sine values are read at the paired *output* coordinate, which is the
    detail that is easy to get wrong in a hand-written backward.
    """
    pid = tl.program_id(0)
    tokens_per_batch = n_heads * seq_len
    batch = pid // tokens_per_batch
    rem = pid % tokens_per_batch
    head = rem // seq_len
    seq = rem % seq_len

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim
    half = head_dim // 2
    paired = tl.where(offsets < half, offsets + half, offsets - half)
    inverse_sign = tl.where(offsets < half, 1.0, -1.0)

    x_base = ((batch * n_heads + head) * seq_len + seq) * head_dim
    pos_base = (batch * seq_len + seq) * head_dim

    grad_y = tl.load(grad_y_ptr + x_base + offsets, mask=mask, other=0.0).to(tl.float32)
    paired_grad_y = tl.load(
        grad_y_ptr + x_base + paired, mask=mask, other=0.0
    ).to(tl.float32)
    cos_value = tl.load(cos_ptr + pos_base + offsets, mask=mask, other=0.0).to(tl.float32)
    paired_sin_value = tl.load(
        sin_ptr + pos_base + paired, mask=mask, other=0.0
    ).to(tl.float32)

    # First undo RoPE.  This is the gradient with respect to the RMSNorm
    # output, not yet the gradient with respect to the original input.
    grad_rope = (
        grad_y * cos_value
        + inverse_sign * paired_grad_y * paired_sin_value
    )

    # Then apply the RMSNorm backward formula.
    x = tl.load(
        x_ptr + x_base + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(
        w_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / head_dim
    inv_rms = tl.rsqrt(mean_square + eps)

    weighted_grad = grad_rope * weight
    correction = tl.sum(weighted_grad * x, axis=0) / head_dim
    grad_x = (
        weighted_grad - x * inv_rms * inv_rms * correction
    ) * inv_rms

    tl.store(grad_x_ptr + x_base + offsets, grad_x, mask=mask)

    # This reduction spans batch/head/sequence programs, hence atomic_add.
    grad_weight = grad_rope * x * inv_rms
    tl.atomic_add(grad_w_ptr + offsets, grad_weight, mask=mask)


def _fused_backward_one(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch fused backward for one tensor laid out as [B,S,H,D]."""
    batch, seq_len, heads, head_dim = x.shape
    x_kernel = x.transpose(1, 2).contiguous()
    grad_y_kernel = grad_y.transpose(1, 2).contiguous()
    grad_x_kernel = torch.empty_like(x_kernel)
    # Triton 3.1 does not support atomic_add on BF16. Accumulate the
    # reduction in FP32 and cast back to the parameter dtype afterwards.
    grad_weight_accum = torch.zeros(
        head_dim,
        device=weight.device,
        dtype=torch.float32,
    )
    cos = cos.contiguous()
    sin = sin.contiguous()
    weight = weight.contiguous()
    block_size = triton.next_power_of_2(head_dim)
    _rope_backward_kernel[(batch * heads * seq_len,)](
        x_kernel,
        grad_y_kernel,
        cos,
        sin,
        weight,
        grad_x_kernel,
        grad_weight_accum,
        heads,
        seq_len,
        head_dim,
        eps,
        BLOCK_SIZE=block_size,
    )
    return (
        grad_x_kernel.transpose(1, 2).contiguous(),
        grad_weight_accum.to(dtype=weight.dtype),
    )




class _FusedQKNormRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, q_weight, k_weight, cos, sin, eps):
        ctx.save_for_backward(q, k, q_weight, k_weight, cos, sin)
        ctx.eps = eps
        return (
            _fused_one(q, q_weight, cos, sin, eps),
            _fused_one(k, k_weight, cos, sin, eps),
        )

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        q, k, q_weight, k_weight, cos, sin = ctx.saved_tensors
        grad_q_x = grad_q_qw = None
        grad_k_x = grad_k_kw = None
        if grad_q is not None:
            grad_q_x, grad_q_qw = _fused_backward_one(
                q, q_weight, grad_q, cos, sin, ctx.eps
            )
        if grad_k is not None:
            grad_k_x, grad_k_kw = _fused_backward_one(
                k, k_weight, grad_k, cos, sin, ctx.eps
            )
        return grad_q_x, grad_k_x, grad_q_qw, grad_k_kw, None, None, None


def fused_qk_norm_rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fused Q/K RMSNorm + RoPE, launching one kernel per tensor."""
    return _FusedQKNormRoPEFunction.apply(q, k, q_weight, k_weight, cos, sin, eps)
