"""Correctness-first Triton backward kernels for causal GQA attention.

The three kernels are deliberately independent: dQ, dK, and dV.  They
recompute tile probabilities from Q/K and a saved log-sum-exp tensor instead
of materializing the full attention probability matrix.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


BLOCK_M = 16
BLOCK_N = 64


@triton.jit
def _attention_dq_partial_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    dq_partial_ptr,
    q_heads,
    kv_heads,
    seq_len,
    head_dim,
    num_q_blocks,
    num_k_blocks,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute one dQ contribution for one query/key tile pair.

    MODIFIED: dQ is materialized as tile partials in this correctness-first
    version.  The host sums the key-tile dimension after the kernel, avoiding
    the multi-iteration dQ accumulation path until its optimized form is
    validated.
    """
    pid = tl.program_id(0)
    batch = pid // (q_heads * num_q_blocks * num_k_blocks)
    rem = pid % (q_heads * num_q_blocks * num_k_blocks)
    q_head = rem // (num_q_blocks * num_k_blocks)
    rem = rem % (num_q_blocks * num_k_blocks)
    q_block = rem // num_k_blocks
    k_block = rem % num_k_blocks
    kv_head = q_head // (q_heads // kv_heads)

    q_pos = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_pos = k_block * BLOCK_N + tl.arange(0, BLOCK_N)
    q_mask = q_pos < seq_len
    key_mask = key_pos < seq_len
    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim
    q_base = ((batch * q_heads + q_head) * seq_len) * head_dim
    q_off = q_base + q_pos[:, None] * head_dim + d[None, :]
    q = tl.load(q_ptr + q_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptr + q_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    lse = tl.load(lse_ptr + (batch * q_heads + q_head) * seq_len + q_pos, mask=q_mask, other=0.0)
    kv_base = (batch * kv_heads + kv_head) * seq_len * head_dim
    k_off = kv_base + key_pos[:, None] * head_dim + d[None, :]
    k = tl.load(k_ptr + k_off, mask=key_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    v = tl.load(v_ptr + k_off, mask=key_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    valid = q_mask[:, None] & key_mask[None, :] & (key_pos[None, :] <= q_pos[:, None])
    scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale
    masked_scores = tl.where(valid, scores, -float("inf"))
    probs = tl.where(valid, tl.exp(masked_scores - lse[:, None]), 0.0)
    dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")
    dp_dot_p = tl.sum(dp * probs, axis=1)
    ds = probs * (dp - dp_dot_p[:, None])
    dq = tl.sum(ds[:, :, None] * k[None, :, :], axis=1) * scale

    partial_base = (
        (((batch * q_heads + q_head) * num_q_blocks + q_block) * num_k_blocks + k_block)
        * BLOCK_M * head_dim
    )
    partial_off = partial_base + tl.arange(0, BLOCK_M)[:, None] * head_dim + d[None, :]
    tl.store(dq_partial_ptr + partial_off, dq, mask=q_mask[:, None] & d_mask[None, :])


@triton.jit
def _attention_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    dq_ptr,
    q_heads,
    kv_heads,
    seq_len,
    head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
):
    """Compute dQ for one (batch, query-head, query-tile) program."""
    pid = tl.program_id(0)
    num_q_blocks = tl.cdiv(seq_len, BLOCK_M)
    batch = pid // (q_heads * num_q_blocks)
    rem = pid % (q_heads * num_q_blocks)
    q_head = rem // num_q_blocks
    q_block = rem % num_q_blocks
    kv_head = q_head // (q_heads // kv_heads)

    q_pos = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_pos < seq_len
    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim
    q_base = ((batch * q_heads + q_head) * seq_len) * head_dim
    q_off = q_base + q_pos[:, None] * head_dim + d[None, :]
    q = tl.load(q_ptr + q_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptr + q_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    lse = tl.load(lse_ptr + (batch * q_heads + q_head) * seq_len + q_pos, mask=q_mask, other=0.0)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    kv_base = (batch * kv_heads + kv_head) * seq_len * head_dim
    for key_start in range(0, MAX_SEQ_LEN, BLOCK_N):
        key_pos = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_pos < seq_len
        valid = q_mask[:, None] & key_mask[None, :] & (key_pos[None, :] <= q_pos[:, None])
        active_rows = q_pos >= key_start
        k_off = kv_base + key_pos[:, None] * head_dim + d[None, :]
        v_off = k_off
        k = tl.load(k_ptr + k_off, mask=key_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + v_off, mask=key_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        q_active = tl.where(active_rows[:, None], q, 0.0)
        do_active = tl.where(active_rows[:, None], do, 0.0)
        scores = tl.dot(q_active, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale
        # Explicitly zero invalid entries before exponentiation and dot
        # products.  This is
        # important for key tiles that lie entirely to the right of a causal
        # query row (especially when seq_len is not a multiple of BLOCK_N).
        masked_scores = tl.where(valid, scores, -float("inf"))
        probs = tl.where(valid, tl.exp(masked_scores - lse[:, None]), 0.0)
        dp = tl.dot(do_active, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")
        dp_dot_p = tl.sum(dp * probs, axis=1)
        ds = probs * (dp - dp_dot_p[:, None])
        # Correctness-first reduction.  This avoids relying on a dot layout
        # for the dQ accumulation while we validate the three-kernel path.
        dq_contrib = tl.sum(ds[:, :, None] * k[None, :, :], axis=1) * scale
        dq += tl.where(active_rows[:, None], dq_contrib, 0.0)

    tl.store(dq_ptr + q_off, dq, mask=q_mask[:, None] & d_mask[None, :])


@triton.jit
def _attention_dk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    dk_ptr,
    q_heads,
    kv_heads,
    seq_len,
    head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
):
    """Compute and atomically accumulate dK for one Q/K tile pair."""
    pid = tl.program_id(0)
    num_q_blocks = tl.cdiv(seq_len, BLOCK_M)
    num_k_blocks = tl.cdiv(seq_len, BLOCK_N)
    batch = pid // (q_heads * num_q_blocks * num_k_blocks)
    rem = pid % (q_heads * num_q_blocks * num_k_blocks)
    q_head = rem // (num_q_blocks * num_k_blocks)
    rem = rem % (num_q_blocks * num_k_blocks)
    q_block = rem // num_k_blocks
    k_block = rem % num_k_blocks
    kv_head = q_head // (q_heads // kv_heads)

    q_pos = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_pos = k_block * BLOCK_N + tl.arange(0, BLOCK_N)
    q_mask = q_pos < seq_len
    key_mask = key_pos < seq_len
    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim
    q_base = ((batch * q_heads + q_head) * seq_len) * head_dim
    q_off = q_base + q_pos[:, None] * head_dim + d[None, :]
    q_raw = tl.load(q_ptr + q_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    do_raw = tl.load(do_ptr + q_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    q = q_raw.to(tl.float32)
    do = do_raw.to(tl.float32)
    lse = tl.load(lse_ptr + (batch * q_heads + q_head) * seq_len + q_pos, mask=q_mask, other=0.0)
    kv_base = (batch * kv_heads + kv_head) * seq_len * head_dim
    k_off = kv_base + key_pos[:, None] * head_dim + d[None, :]
    k = tl.load(k_ptr + k_off, mask=key_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    v = tl.load(v_ptr + k_off, mask=key_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    valid = q_mask[:, None] & key_mask[None, :] & (key_pos[None, :] <= q_pos[:, None])
    scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale
    masked_scores = tl.where(valid, scores, -float("inf"))
    probs = tl.where(valid, tl.exp(masked_scores - lse[:, None]), 0.0)
    dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")
    dp_dot_p = tl.sum(dp * probs, axis=1)
    ds = probs * (dp - dp_dot_p[:, None])
    # Correctness-first reduction over the BLOCK_M query rows.
    dk = tl.sum(ds[:, :, None] * q[:, None, :], axis=0) * scale
    tl.atomic_add(dk_ptr + k_off, dk, mask=key_mask[:, None] & d_mask[None, :])


@triton.jit
def _attention_dv_kernel(
    q_ptr,
    k_ptr,
    do_ptr,
    lse_ptr,
    dv_ptr,
    q_heads,
    kv_heads,
    seq_len,
    head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
):
    """Compute and atomically accumulate dV for one Q/K tile pair."""
    pid = tl.program_id(0)
    num_q_blocks = tl.cdiv(seq_len, BLOCK_M)
    num_k_blocks = tl.cdiv(seq_len, BLOCK_N)
    batch = pid // (q_heads * num_q_blocks * num_k_blocks)
    rem = pid % (q_heads * num_q_blocks * num_k_blocks)
    q_head = rem // (num_q_blocks * num_k_blocks)
    rem = rem % (num_q_blocks * num_k_blocks)
    q_block = rem // num_k_blocks
    k_block = rem % num_k_blocks
    kv_head = q_head // (q_heads // kv_heads)

    q_pos = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_pos = k_block * BLOCK_N + tl.arange(0, BLOCK_N)
    q_mask = q_pos < seq_len
    key_mask = key_pos < seq_len
    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim
    q_base = ((batch * q_heads + q_head) * seq_len) * head_dim
    q_off = q_base + q_pos[:, None] * head_dim + d[None, :]
    q_raw = tl.load(q_ptr + q_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    do_raw = tl.load(do_ptr + q_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    q = q_raw.to(tl.float32)
    do = do_raw.to(tl.float32)
    lse = tl.load(lse_ptr + (batch * q_heads + q_head) * seq_len + q_pos, mask=q_mask, other=0.0)
    kv_head = q_head // (q_heads // kv_heads)
    kv_base = (batch * kv_heads + kv_head) * seq_len * head_dim
    k_off = kv_base + key_pos[:, None] * head_dim + d[None, :]
    k = tl.load(k_ptr + k_off, mask=key_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    valid = q_mask[:, None] & key_mask[None, :] & (key_pos[None, :] <= q_pos[:, None])
    scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale
    masked_scores = tl.where(valid, scores, -float("inf"))
    probs = tl.where(valid, tl.exp(masked_scores - lse[:, None]), 0.0)
    # Correctness-first reduction for dV; a later optimized version can use
    # a matrix dot once the indexing and masking contract is frozen.
    dv = tl.sum(probs[:, :, None] * do[:, None, :], axis=0)
    tl.atomic_add(dv_ptr + k_off, dv, mask=key_mask[:, None] & d_mask[None, :])


@triton.jit
def _attention_dkdv_fused_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    dk_ptr,
    dv_ptr,
    q_heads,
    kv_heads,
    seq_len,
    head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused dK+dV for one query tile and one key/value tile.

    The probability and softmax-backward intermediate are computed once and
    shared by both gradient paths.  Multiple query programs contribute to the
    same KV gradient, so the final stores are FP32 atomic additions.
    """
    pid = tl.program_id(0)
    num_q_blocks = tl.cdiv(seq_len, BLOCK_M)
    num_k_blocks = tl.cdiv(seq_len, BLOCK_N)
    batch = pid // (q_heads * num_q_blocks * num_k_blocks)
    rem = pid % (q_heads * num_q_blocks * num_k_blocks)
    q_head = rem // (num_q_blocks * num_k_blocks)
    rem = rem % (num_q_blocks * num_k_blocks)
    q_block = rem // num_k_blocks
    k_block = rem % num_k_blocks
    kv_head = q_head // (q_heads // kv_heads)

    q_pos = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_pos = k_block * BLOCK_N + tl.arange(0, BLOCK_N)
    q_mask = q_pos < seq_len
    key_mask = key_pos < seq_len
    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim

    q_base = ((batch * q_heads + q_head) * seq_len) * head_dim
    q_off = q_base + q_pos[:, None] * head_dim + d[None, :]
    q = tl.load(q_ptr + q_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptr + q_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    lse = tl.load(
        lse_ptr + (batch * q_heads + q_head) * seq_len + q_pos,
        mask=q_mask,
        other=0.0,
    )

    kv_base = (batch * kv_heads + kv_head) * seq_len * head_dim
    kv_off = kv_base + key_pos[:, None] * head_dim + d[None, :]
    k = tl.load(k_ptr + kv_off, mask=key_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    v = tl.load(v_ptr + kv_off, mask=key_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    valid = q_mask[:, None] & key_mask[None, :] & (key_pos[None, :] <= q_pos[:, None])
    scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale
    masked_scores = tl.where(valid, scores, -float("inf"))
    probs = tl.where(valid, tl.exp(masked_scores - lse[:, None]), 0.0)
    dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")
    dp_dot_p = tl.sum(dp * probs, axis=1)
    ds = probs * (dp - dp_dot_p[:, None])

    # Both reductions reuse the same probs/ds/Q/dO values.
    dk = tl.sum(ds[:, :, None] * q[:, None, :], axis=0) * scale
    dv = tl.sum(probs[:, :, None] * do[:, None, :], axis=0)
    tl.atomic_add(dk_ptr + kv_off, dk, mask=key_mask[:, None] & d_mask[None, :])
    tl.atomic_add(dv_ptr + kv_off, dv, mask=key_mask[:, None] & d_mask[None, :])


def _compute_lse(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    """Small PyTorch helper for the first correctness version."""
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if q_heads != kv_heads:
        k = k.repeat_interleave(q_heads // kv_heads, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    seq_len = q.shape[-2]
    positions = torch.arange(seq_len, device=q.device)
    scores = scores.masked_fill(positions[None, :] > positions[:, None], float("-inf"))
    return torch.logsumexp(scores, dim=-1)


def _validate_kv_backward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[int, int, int, int]:
    if not all(t.is_cuda for t in (q, k, v, grad_out)):
        raise ValueError("attention backward requires CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors")
    batch, q_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[1]
    if grad_out.shape != q.shape or k.shape != (batch, kv_heads, seq_len, head_dim):
        raise ValueError("invalid q/k/grad_out shapes")
    if v.shape != k.shape or q_heads % kv_heads != 0:
        raise ValueError("invalid GQA configuration or v shape")
    return batch, q_heads, kv_heads, seq_len


def attention_kv_backward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    fused: bool,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Benchmark-only dK/dV path, excluding dQ and host reductions."""
    batch, q_heads, kv_heads, seq_len = _validate_kv_backward_inputs(q, k, v, grad_out)
    head_dim = q.shape[-1]
    q, k, v, grad_out = (t.contiguous() for t in (q, k, v, grad_out))
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    block_d = triton.next_power_of_2(head_dim)
    block_n = 32 if block_d >= 128 else BLOCK_N
    num_q_blocks = triton.cdiv(seq_len, BLOCK_M)
    num_k_blocks = triton.cdiv(seq_len, block_n)
    lse = _compute_lse(q, k, scale).contiguous()
    dk = torch.zeros((batch, kv_heads, seq_len, head_dim), device=q.device, dtype=torch.float32)
    dv = torch.zeros_like(dk)
    grid = (batch * q_heads * num_q_blocks * num_k_blocks,)
    if fused:
        _attention_dkdv_fused_kernel[grid](
            q, k, v, grad_out, lse, dk, dv,
            q_heads, kv_heads, seq_len, head_dim, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=block_d,
            num_warps=4, num_stages=1,
        )
    else:
        _attention_dk_kernel[grid](
            q, k, v, grad_out, lse, dk,
            q_heads, kv_heads, seq_len, head_dim, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=block_d, MAX_SEQ_LEN=seq_len,
            num_warps=4, num_stages=1,
        )
        _attention_dv_kernel[grid](
            q, k, grad_out, lse, dv,
            q_heads, kv_heads, seq_len, head_dim, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=block_d, MAX_SEQ_LEN=seq_len,
            num_warps=4, num_stages=1,
        )
    return dk.to(k.dtype), dv.to(v.dtype)


def attention_backward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    scale: float | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch independent Triton dQ/dK/dV kernels for causal GQA attention."""
    if not all(t.is_cuda for t in (q, k, v, grad_out)):
        raise ValueError("attention_backward_triton requires CUDA tensors")
    batch, q_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[1]
    if grad_out.shape != q.shape or v.shape != k.shape:
        raise ValueError("expected grad_out like q and v like k")
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    q, k, v, grad_out = (t.contiguous() for t in (q, k, v, grad_out))
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    block_d = triton.next_power_of_2(head_dim)
    # MODIFIED: D=128 with BLOCK_N=64 can exceed the A6000 shared-memory
    # budget in this correctness-first kernel, so use a smaller K/V tile.
    block_n = 32 if block_d >= 128 else BLOCK_N
    if lse is None:
        lse = _compute_lse(q, k, scale).contiguous()
    else:
        lse = lse.contiguous()
    # MODIFIED: keep one dQ partial per key tile, then reduce on the host.
    num_q_blocks = triton.cdiv(seq_len, BLOCK_M)
    num_k_blocks = triton.cdiv(seq_len, block_n)
    dq_partial = torch.zeros(
        (batch, q_heads, num_q_blocks, num_k_blocks, BLOCK_M, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    dk_acc = torch.zeros((batch, kv_heads, seq_len, head_dim), device=q.device, dtype=torch.float32)
    dv_acc = torch.zeros_like(dk_acc)
    q_grid = (batch * q_heads * num_q_blocks * num_k_blocks,)
    kv_grid = (batch * q_heads * num_q_blocks * num_k_blocks,)
    common = (q_heads, kv_heads, seq_len, head_dim, scale)
    _attention_dq_partial_kernel[q_grid](
        q, k, v, grad_out, lse, dq_partial,
        q_heads, kv_heads, seq_len, head_dim, num_q_blocks, num_k_blocks, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=block_d,
        num_warps=4, num_stages=1,
    )
    _attention_dk_kernel[kv_grid](
        q, k, v, grad_out, lse, dk_acc, *common,
        BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=block_d, MAX_SEQ_LEN=seq_len,
        num_warps=4, num_stages=1,
    )
    _attention_dv_kernel[kv_grid](
        q, k, grad_out, lse, dv_acc, *common,
        BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=block_d, MAX_SEQ_LEN=seq_len,
        num_warps=4, num_stages=1,
    )
    dq = dq_partial.sum(dim=3).reshape(batch, q_heads, num_q_blocks * BLOCK_M, head_dim)[..., :seq_len, :]
    return dq.to(dtype=q.dtype), dk_acc.to(dtype=k.dtype), dv_acc.to(dtype=v.dtype)


def attention_backward_triton_fused_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    scale: float | None = None,
    lse: torch.Tensor | None = None,
    fused_dq: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward path with fused dK+dV and optional register-accumulated dQ.

    With ``fused_dq=True`` one program owns a query tile, walks all key
    tiles, accumulates dQ on chip, and stores it once.  ``False`` retains the
    partial-workspace implementation as a baseline.
    """
    if not all(t.is_cuda for t in (q, k, v, grad_out)):
        raise ValueError("attention_backward_triton requires CUDA tensors")
    batch, q_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[1]
    q, k, v, grad_out = (t.contiguous() for t in (q, k, v, grad_out))
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    block_d = triton.next_power_of_2(head_dim)
    block_n = 32 if block_d >= 128 else BLOCK_N
    num_q_blocks = triton.cdiv(seq_len, BLOCK_M)
    num_k_blocks = triton.cdiv(seq_len, block_n)
    if lse is None:
        lse = _compute_lse(q, k, scale).contiguous()
    else:
        lse = lse.contiguous()
    if fused_dq:
        dq_acc = torch.empty(
            (batch, q_heads, seq_len, head_dim), device=q.device, dtype=torch.float32
        )
    else:
        dq_partial = torch.zeros(
            (batch, q_heads, num_q_blocks, num_k_blocks, BLOCK_M, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
    dk_acc = torch.zeros((batch, kv_heads, seq_len, head_dim), device=q.device, dtype=torch.float32)
    dv_acc = torch.zeros_like(dk_acc)
    if fused_dq:
        dq_grid = (batch * q_heads * num_q_blocks,)
        _attention_dq_kernel[dq_grid](
            q, k, v, grad_out, lse, dq_acc,
            q_heads, kv_heads, seq_len, head_dim, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=block_d,
            MAX_SEQ_LEN=seq_len,
            num_warps=4, num_stages=1,
        )
    else:
        dq_grid = (batch * q_heads * num_q_blocks * num_k_blocks,)
        _attention_dq_partial_kernel[dq_grid](
            q, k, v, grad_out, lse, dq_partial,
            q_heads, kv_heads, seq_len, head_dim, num_q_blocks, num_k_blocks, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=block_d,
            num_warps=4, num_stages=1,
        )
    dkdv_grid = (batch * q_heads * num_q_blocks * num_k_blocks,)
    _attention_dkdv_fused_kernel[dkdv_grid](
        q, k, v, grad_out, lse, dk_acc, dv_acc,
        q_heads, kv_heads, seq_len, head_dim, scale,
        BLOCK_D=block_d,
        BLOCK_M=BLOCK_M,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )
    if fused_dq:
        dq = dq_acc
    else:
        dq = dq_partial.sum(dim=3).reshape(batch, q_heads, num_q_blocks * BLOCK_M, head_dim)[..., :seq_len, :]
    return dq.to(q.dtype), dk_acc.to(k.dtype), dv_acc.to(v.dtype)
