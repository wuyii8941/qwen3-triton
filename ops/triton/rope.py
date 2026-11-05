"""Standalone Qwen3 RoPE Triton forward and backward kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    y_ptr,
    n_heads,
    seq_len,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
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
    pos_base = (batch * seq_len + seq) * head_dim
    x_value = tl.load(x_ptr + x_base + offsets, mask=mask, other=0.0).to(tl.float32)
    paired_value = tl.load(x_ptr + x_base + paired, mask=mask, other=0.0).to(tl.float32)
    cos_value = tl.load(cos_ptr + pos_base + offsets, mask=mask, other=0.0).to(tl.float32)
    sin_value = tl.load(sin_ptr + pos_base + offsets, mask=mask, other=0.0).to(tl.float32)
    rotated = x_value * cos_value + (paired_value * sign) * sin_value
    tl.store(y_ptr + x_base + offsets, rotated, mask=mask)


@triton.jit
def _rope_backward_kernel(
    grad_y_ptr,
    cos_ptr,
    sin_ptr,
    grad_x_ptr,
    n_heads,
    seq_len,
    head_dim,
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

    grad_x = grad_y * cos_value + inverse_sign * paired_grad_y * paired_sin_value
    tl.store(grad_x_ptr + x_base + offsets, grad_x, mask=mask)


def _rope_forward(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not cos.is_cuda or not sin.is_cuda:
        raise ValueError("rope_triton requires CUDA tensors")
    if x.ndim != 4 or cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("expected x=[batch,heads,seq,head_dim] and cos/sin=[batch,seq,head_dim]")
    batch, heads, seq_len, head_dim = x.shape
    if cos.shape != (batch, seq_len, head_dim):
        raise ValueError("cos/sin shape must match x batch, sequence, and head_dim")
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(head_dim)
    if block_size > 65536:
        raise ValueError("head_dim is too large for this first kernel")
    _rope_kernel[(batch * heads * seq_len,)](
        x,
        cos,
        sin,
        output,
        heads,
        seq_len,
        head_dim,
        BLOCK_SIZE=block_size,
    )
    return output


def _rope_backward(grad_y: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Launch the Triton RoPE backward for one tensor (q or k)."""
    if not grad_y.is_cuda or not cos.is_cuda or not sin.is_cuda:
        raise ValueError("rope_triton requires CUDA tensors")
    if grad_y.ndim != 4 or cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("expected grad=[batch,heads,seq,head_dim] and cos/sin=[batch,seq,head_dim]")
    batch, heads, seq_len, head_dim = grad_y.shape
    if cos.shape != (batch, seq_len, head_dim):
        raise ValueError("cos/sin shape must match grad batch, sequence, and head_dim")
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    grad_y = grad_y.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    grad_x = torch.empty_like(grad_y)
    block_size = triton.next_power_of_2(head_dim)
    if block_size > 65536:
        raise ValueError("head_dim is too large for this first kernel")
    _rope_backward_kernel[(batch * heads * seq_len,)](
        grad_y,
        cos,
        sin,
        grad_x,
        heads,
        seq_len,
        head_dim,
        BLOCK_SIZE=block_size,
    )
    return grad_x


class _RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _rope_forward(q, cos, sin), _rope_forward(k, cos, sin)

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        grad_q_input = _rope_backward(grad_q, cos, sin) if grad_q is not None else None
        grad_k_input = _rope_backward(grad_k, cos, sin) if grad_k is not None else None
        return grad_q_input, grad_k_input, None, None


def apply_rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _RoPEFunction.apply(q, k, cos, sin)
