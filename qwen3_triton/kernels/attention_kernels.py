"""Explicit Triton attention kernels adapted from this repository's existing operators.

Forward uses online softmax; backward uses deterministic KV ownership and GQA
head reduction. No ATen fallback or floating-point atomic accumulation.
"""
import triton
import triton.language as tl

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
    DOT_PRECISION: tl.constexpr,
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
    for key_start in range(0, tl.minimum(MAX_SEQ_LEN, (q_block + 1) * BLOCK_M), BLOCK_N):
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
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision=DOT_PRECISION) * scale
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
        acc = acc * old_scale[:, None] + tl.dot(probs, v, input_precision=DOT_PRECISION)
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

@triton.jit
def _attention_delta_kernel(output_ptr, grad_ptr, delta_ptr, head_dim: tl.constexpr,
                            BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    output = tl.load(output_ptr + row * head_dim + d, d < head_dim, other=0.).to(tl.float32)
    grad = tl.load(grad_ptr + row * head_dim + d, d < head_dim, other=0.).to(tl.float32)
    tl.store(delta_ptr + row, tl.sum(output * grad, axis=0))

@triton.jit
def _attention_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
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
    DOT_PRECISION: tl.constexpr,
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
    for key_start in range(0, tl.minimum(MAX_SEQ_LEN, (q_block + 1) * BLOCK_M), BLOCK_N):
        key_pos = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_pos < seq_len
        valid = q_mask[:, None] & key_mask[None, :] & (key_pos[None, :] <= q_pos[:, None])
        k_off = kv_base + key_pos[:, None] * head_dim + d[None, :]
        v_off = k_off
        k = tl.load(k_ptr + k_off, mask=key_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + v_off, mask=key_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision=DOT_PRECISION) * scale
        # Explicitly zero invalid entries before exponentiation and dot
        # products.  This is
        # important for key tiles that lie entirely to the right of a causal
        # query row (especially when seq_len is not a multiple of BLOCK_N).
        masked_scores = tl.where(valid, scores, -float("inf"))
        probs = tl.where(valid, tl.exp(masked_scores - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision=DOT_PRECISION)
        dp_dot_p = tl.load(delta_ptr + (batch * q_heads + q_head) * seq_len + q_pos, mask=q_mask, other=0.0)
        ds = probs * (dp - dp_dot_p[:, None])
        dq_contrib = tl.dot(ds, k, input_precision=DOT_PRECISION) * scale
        dq += dq_contrib

    tl.store(dq_ptr + q_off, dq, mask=q_mask[:, None] & d_mask[None, :])

@triton.jit
def _attention_dkdv_fused_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dk_ptr, dv_ptr,
    q_heads: tl.constexpr, kv_heads: tl.constexpr, seq_len: tl.constexpr,
    head_dim: tl.constexpr, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    n_key_blocks = tl.cdiv(seq_len, BLOCK_N)
    batch = pid // (q_heads * n_key_blocks)
    head = (pid // n_key_blocks) % q_heads
    key_block = pid % n_key_blocks
    kv_head = head // (q_heads // kv_heads)
    keys = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    kv_offsets = ((batch * kv_heads + kv_head) * seq_len + keys[:, None]) * head_dim + dims[None, :]
    kv_mask = (keys[:, None] < seq_len) & (dims[None, :] < head_dim)
    k = tl.load(k_ptr + kv_offsets, kv_mask, 0).to(tl.float32)
    v = tl.load(v_ptr + kv_offsets, kv_mask, 0).to(tl.float32)
    dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    # Earlier query rows are causally unable to contribute to this KV tile.
    for start in range(key_block * BLOCK_N // BLOCK_M * BLOCK_M, seq_len, BLOCK_M):
        queries = start + tl.arange(0, BLOCK_M)
        q_offsets = ((batch * q_heads + head) * seq_len + queries[:, None]) * head_dim + dims[None, :]
        q_mask = (queries[:, None] < seq_len) & (dims[None, :] < head_dim)
        q = tl.load(q_ptr + q_offsets, q_mask, 0).to(tl.float32)
        do = tl.load(do_ptr + q_offsets, q_mask, 0).to(tl.float32)
        row_offsets = (batch * q_heads + head) * seq_len + queries
        lse = tl.load(lse_ptr + row_offsets, queries < seq_len, 0)
        delta = tl.load(delta_ptr + row_offsets, queries < seq_len, 0)
        scores = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION) * scale
        valid = ((queries[:, None] < seq_len) & (keys[None, :] < seq_len)
                 & (queries[:, None] >= keys[None, :]))
        probs = tl.where(valid, tl.exp(scores - lse[:, None]), 0.)
        dp = tl.dot(do, tl.trans(v), input_precision=DOT_PRECISION)
        ds = probs * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds), q, input_precision=DOT_PRECISION) * scale
        dv += tl.dot(tl.trans(probs), do, input_precision=DOT_PRECISION)
    # Separate query-head partials avoid GQA atomics too.
    out_offsets = ((batch * q_heads + head) * seq_len + keys[:, None]) * head_dim + dims[None, :]
    tl.store(dk_ptr + out_offsets, dk, kv_mask)
    tl.store(dv_ptr + out_offsets, dv, kv_mask)

@triton.jit
def _attention_kv_head_reduce_kernel(
    partial_k_ptr, partial_v_ptr, dk_ptr, dv_ptr,
    n_elements: tl.constexpr, group_size: tl.constexpr,
    seq_len: tl.constexpr, head_dim: tl.constexpr, kv_heads: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = offsets // (kv_heads * seq_len * head_dim)
    kv_head = (offsets // (seq_len * head_dim)) % kv_heads
    token_dim = offsets % (seq_len * head_dim)
    dk = tl.full((BLOCK,), 0, tl.float32)
    dv = tl.full((BLOCK,), 0, tl.float32)
    for group in range(group_size):
        indices = ((batch * kv_heads + kv_head) * group_size + group) * seq_len * head_dim + token_dim
        dk += tl.load(partial_k_ptr + indices, offsets < n_elements, 0)
        dv += tl.load(partial_v_ptr + indices, offsets < n_elements, 0)
    tl.store(dk_ptr + offsets, dk, offsets < n_elements)
    tl.store(dv_ptr + offsets, dv, offsets < n_elements)
