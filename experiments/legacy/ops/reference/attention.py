"""Readable PyTorch reference for causal scaled dot-product attention."""

from __future__ import annotations

import math

import torch


def scaled_dot_product_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Compute attention explicitly with PyTorch tensor operations.

    Args:
        q: ``[batch, query_heads, query_len, head_dim]``.
        k/v: ``[batch, kv_heads, key_len, head_dim]``.
        is_causal: Apply the usual lower-triangular causal mask.
        enable_gqa: Repeat K/V heads to match Q heads. This is a readable
            reference implementation; a later Triton kernel should map a Q
            head to its KV head without materializing these copies.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors")
    batch, q_heads, query_len, head_dim = q.shape
    k_batch, kv_heads, key_len, k_dim = k.shape
    if (k_batch, k_dim) != (batch, head_dim):
        raise ValueError("q and k batch/head_dim dimensions must match")
    if v.shape != (batch, kv_heads, key_len, head_dim):
        raise ValueError("v must have shape [batch, kv_heads, key_len, head_dim]")
    if q_heads != kv_heads:
        if not enable_gqa:
            raise ValueError("q_heads must equal kv_heads unless enable_gqa=True")
        if q_heads % kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads for GQA")
        groups = q_heads // kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    if is_causal:
        # For standard self-attention query_len == key_len. The offset form
        # also handles a query suffix attending to a longer KV sequence.
        q_positions = torch.arange(query_len, device=q.device) + (key_len - query_len)
        k_positions = torch.arange(key_len, device=q.device)
        allowed = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def scaled_dot_product_attention_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    is_causal: bool = True,
    enable_gqa: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit PyTorch backward oracle for causal attention.

    This intentionally does not call ``autograd.grad``.  The equations are:

    ``dV = P.T @ dO``
    ``dP = dO @ V.T``
    ``dS = P * (dP - sum(dP * P))``
    ``dQ = scale * dS @ K`` and ``dK = scale * dS.T @ Q``.

    For GQA, the expanded K/V gradients are summed back into the original
    ``kv_heads`` layout.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or grad_out.shape != q.shape:
        raise ValueError("invalid attention or grad_out shapes")
    batch, q_heads, query_len, head_dim = q.shape
    _, kv_heads, key_len, _ = k.shape
    if v.shape != (batch, kv_heads, key_len, head_dim):
        raise ValueError("v must have shape [batch, kv_heads, key_len, head_dim]")

    if q_heads != kv_heads:
        if not enable_gqa or q_heads % kv_heads != 0:
            raise ValueError("invalid GQA head configuration")
        groups = q_heads // kv_heads
        k_expanded = k.repeat_interleave(groups, dim=1)
        v_expanded = v.repeat_interleave(groups, dim=1)
    else:
        groups = 1
        k_expanded = k
        v_expanded = v

    scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
    if is_causal:
        q_positions = torch.arange(query_len, device=q.device) + (key_len - query_len)
        k_positions = torch.arange(key_len, device=q.device)
        allowed = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1)

    grad_v_expanded = torch.matmul(probabilities.transpose(-2, -1), grad_out)
    grad_p = torch.matmul(grad_out, v_expanded.transpose(-2, -1))
    softmax_dot = (grad_p * probabilities).sum(dim=-1, keepdim=True)
    grad_scores = probabilities * (grad_p - softmax_dot)
    grad_q = torch.matmul(grad_scores, k_expanded) * scale
    grad_k_expanded = torch.matmul(grad_scores.transpose(-2, -1), q) * scale

    if groups > 1:
        grad_k = grad_k_expanded.view(batch, kv_heads, groups, key_len, head_dim).sum(dim=2)
        grad_v = grad_v_expanded.view(batch, kv_heads, groups, key_len, head_dim).sum(dim=2)
    else:
        grad_k = grad_k_expanded
        grad_v = grad_v_expanded
    return (
        grad_q.to(dtype=q.dtype),
        grad_k.to(dtype=k.dtype),
        grad_v.to(dtype=v.dtype),
    )
