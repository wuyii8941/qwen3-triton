"""Current causal GQA training operator: tiled forward and autograd bridge.

Decode-only implementations are archived under experiments/legacy/ops/triton.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    # tune query-tile size, key-tile size, and warp count instead
    # of hard-coding the one-query-row launch configuration.
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8),
    ],
    # different sequence lengths/head dimensions can prefer
    # different tile configurations.
    key=["seq_len", "head_dim"],
)
@triton.jit
def _attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    q_heads,
    kv_heads,
    seq_len,
    head_dim,
    scale,
    # BLOCK_M lets one program process multiple query rows.
    BLOCK_M: tl.constexpr,
    # BLOCK_N is the streamed K/V tile size.
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
):
    """Causal GQA attention for contiguous [B, H, S, D] tensors."""
    pid = tl.program_id(0)
    # the grid is now one program per (batch, q_head, query tile),
    # rather than one program per individual query row.
    num_query_blocks = tl.cdiv(seq_len, BLOCK_M)
    rows_per_batch = q_heads * num_query_blocks
    batch = pid // rows_per_batch
    rem = pid % rows_per_batch
    q_head = rem // num_query_blocks
    q_block = rem % num_query_blocks

    # GQA mapping without materializing repeated K/V tensors.
    kv_group = q_heads // kv_heads
    kv_head = q_head // kv_group

    # query positions are a BLOCK_M vector, so Q is [BLOCK_M, D].
    q_positions = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    query_mask = q_positions < seq_len
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim
    q_base = ((batch * q_heads + q_head) * seq_len) * head_dim
    q_offsets = q_base + q_positions[:, None] * head_dim + d_offsets[None, :]
    q = tl.load(
        q_ptr + q_offsets,
        mask=query_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    # keep Q in the input dtype (BF16 on the target workload) so
    # the QK matmul can use Tensor Cores; scores are accumulated in FP32.

    # online-softmax state is now one value per query row and the
    # accumulator is [BLOCK_M, D].
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # stream over K/V tiles.  No [S, S] score matrix is allocated.
    for key_start in range(0, MAX_SEQ_LEN, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < seq_len
        causal_mask = key_offsets[None, :] <= q_positions[:, None]
        valid = query_mask[:, None] & key_mask[None, :] & causal_mask

        kv_base = (batch * kv_heads + kv_head) * seq_len * head_dim
        k_offsets = kv_base + key_offsets[:, None] * head_dim + d_offsets[None, :]
        k = tl.load(
            k_ptr + k_offsets,
            mask=key_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        # matrix multiplication computes [BLOCK_M, BLOCK_N] scores.
        # preserve Q/K dtype for the dot and request FP32 output.
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        scores = tl.where(valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, block_max)
        old_scale = tl.exp(row_max - new_max)
        # explicitly broadcast the per-query maximum across the
        # BLOCK_N score columns.
        probs = tl.exp(scores - new_max[:, None])

        v_offsets = kv_base + key_offsets[:, None] * head_dim + d_offsets[None, :]
        v = tl.load(
            v_ptr + v_offsets,
            mask=key_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        # accumulate a whole query tile with a matrix product.
        acc = acc * old_scale[:, None] + tl.dot(probs, v)
        row_sum = row_sum * old_scale + tl.sum(probs, axis=1)
        row_max = new_max

    output = acc / row_sum[:, None]
    # store the complete [BLOCK_M, D] output tile.
    out_offsets = q_base + q_positions[:, None] * head_dim + d_offsets[None, :]
    tl.store(
        out_ptr + out_offsets,
        output,
        mask=query_mask[:, None] & d_mask[None, :],
    )
    # save the per-query log-sum-exp while online softmax state is
    # already in registers. Backward can reuse this instead of rebuilding a
    # full QK^T matrix in PyTorch.
    lse = row_max + tl.log(row_sum)
    lse_base = (batch * q_heads + q_head) * seq_len
    tl.store(lse_ptr + lse_base + q_positions, lse, mask=query_mask)


def attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    scale: float | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Forward-only Triton attention for contiguous [B, H, S, D] tensors.

    The first kernel intentionally requires self-attention with equal query and
    key sequence lengths.  GQA is supported by mapping query heads to KV heads
    inside the kernel rather than repeating K/V in Python.
    """
    if not all(t.is_cuda for t in (q, k, v)):
        raise ValueError("attention_triton requires CUDA tensors")
    if not (q.ndim == k.ndim == v.ndim == 4):
        raise ValueError("q, k, and v must be rank-4 tensors")
    batch, q_heads, q_len, head_dim = q.shape
    k_batch, kv_heads, k_len, k_dim = k.shape
    if (k_batch, k_len, k_dim) != (batch, q_len, head_dim):
        raise ValueError("first kernel requires q/k sequence and head_dim shapes to match")
    if v.shape != (batch, kv_heads, q_len, head_dim):
        raise ValueError("v must have shape [batch, kv_heads, seq_len, head_dim]")
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads for GQA")
    if not is_causal:
        raise NotImplementedError("first attention kernel only supports causal attention")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even number")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = torch.empty_like(q)
    # LSE is a compact [B, Hq, S] backward context tensor.
    lse = torch.empty((batch, q_heads, q_len), device=q.device, dtype=torch.float32)
    # BLOCK_D is the compile-time padded head dimension; BLOCK_M and
    # BLOCK_N are selected by the autotuner.
    block_d = triton.next_power_of_2(head_dim)
    if block_d > 256:
        raise ValueError("first attention kernel supports head_dim <= 256")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    # the grid depends on the BLOCK_M selected by autotune.
    grid = lambda meta: (batch * q_heads * triton.cdiv(q_len, meta["BLOCK_M"]),)
    _attention_forward_kernel[grid](
        q,
        k,
        v,
        output,
        lse,
        q_heads,
        kv_heads,
        q_len,
        head_dim,
        scale,
        BLOCK_D=block_d,
        MAX_SEQ_LEN=q_len,
    )
    if return_lse:
        return output, lse
    return output


def attention_triton_qwen_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Qwen prefill entry point for the fixed ``head_dim=128`` workload.

    K/V stay in ``[B, Hkv, S, 128]`` form and are mapped to query heads in
    the Triton kernel; no ``repeat_interleave`` or expanded KV allocation is
    performed.  This explicit entry point lets us tune the Qwen shape
    independently from the generic head-dimension path.
    """
    if q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("Qwen prefill requires q with shape [B, Hq, S, 128]")
    if k.shape[-1] != 128 or v.shape != k.shape:
        raise ValueError("Qwen prefill requires k/v with shape [B, Hkv, S, 128]")
    if q.shape[2] != k.shape[2]:
        raise ValueError("prefill requires equal query and KV sequence lengths")
    return attention_triton(q, k, v, scale=scale, return_lse=return_lse)






def attention_triton_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return forward output plus the compact LSE needed by backward."""
    output, lse = attention_triton(
        q,
        k,
        v,
        is_causal=is_causal,
        scale=scale,
        return_lse=True,
    )
    return output, lse


class _TritonAttentionFunction(torch.autograd.Function):
    """Autograd bridge for the Triton forward and backward implementations."""

    @staticmethod
    def forward(ctx, q, k, v, scale):
        if q.shape[-1] == 128 and q.shape[-2] > 1:
            output, lse = attention_triton_qwen_prefill(
                q, k, v, scale=scale, return_lse=True
            )
        else:
            output, lse = attention_triton_with_lse(q, k, v, scale=scale)
        ctx.save_for_backward(q, k, v, lse)
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Import lazily to keep the standalone forward module lightweight.
        from ops.triton.attention_backward import attention_backward_triton

        q, k, v, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = attention_backward_triton(
            q,
            k,
            v,
            grad_output,
            scale=ctx.scale,
            lse=lse,
        )
        return grad_q, grad_k, grad_v, None


def attention_triton_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Trainable Triton attention using the saved-LSE backward path."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return _TritonAttentionFunction.apply(q, k, v, scale)
