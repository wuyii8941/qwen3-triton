"""Break down the fused attention backward into measurable stages."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.triton.attention_backward import (
    BLOCK_M,
    BLOCK_N,
    _attention_dkdv_fused_kernel,
    _attention_dq_partial_kernel,
    _compute_lse,
)
from ops.triton.attention import attention_triton_with_lse


def measure(fn, warmup: int = 5, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    batch, q_heads, kv_heads, seq_len, head_dim = 1, 16, 8, 128, 128
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, q_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    grad_out = torch.randn_like(q)
    block_d = triton.next_power_of_2(head_dim)
    block_n = 32 if block_d >= 128 else BLOCK_N
    num_q_blocks = triton.cdiv(seq_len, BLOCK_M)
    num_k_blocks = triton.cdiv(seq_len, block_n)
    grid = (batch * q_heads * num_q_blocks * num_k_blocks,)

    # MODIFIED: obtain LSE from the forward Triton kernel. Backward stages use
    # this saved tensor and do not recompute it with PyTorch.
    _, lse = attention_triton_with_lse(q, k, v, scale=scale)
    dq_partial = torch.zeros(
        (batch, q_heads, num_q_blocks, num_k_blocks, BLOCK_M, head_dim),
        device="cuda",
        dtype=torch.float32,
    )
    dk = torch.zeros((batch, kv_heads, seq_len, head_dim), device="cuda", dtype=torch.float32)
    dv = torch.zeros_like(dk)

    def lse_recompute_fallback_stage() -> None:
        _compute_lse(q, k, scale)

    def dq_stage() -> None:
        _attention_dq_partial_kernel[grid](
            q, k, v, grad_out, lse, dq_partial,
            q_heads, kv_heads, seq_len, head_dim, num_q_blocks, num_k_blocks, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=block_d,
            num_warps=4, num_stages=1,
        )

    def dkdv_stage() -> None:
        dk.zero_()
        dv.zero_()
        _attention_dkdv_fused_kernel[grid](
            q, k, v, grad_out, lse, dk, dv,
            q_heads, kv_heads, seq_len, head_dim, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=block_d,
            num_warps=4, num_stages=1,
        )

    def reduction_stage() -> None:
        dq_partial.sum(dim=3)

    results = {
        "lse_recompute_fallback": measure(lse_recompute_fallback_stage),
        "dQ_partial": measure(dq_stage),
        "fused_dK_dV": measure(dkdv_stage),
        "dQ_reduction": measure(reduction_stage),
    }
    backward_saved_lse = results["dQ_partial"] + results["fused_dK_dV"] + results["dQ_reduction"]
    total = sum(results.values())
    print(f"shape: B={batch} Hq={q_heads} Hkv={kv_heads} S={seq_len} D={head_dim} dtype={dtype}")
    for name, value in results.items():
        print(f"{name:16s} {value:8.3f} ms  ({100.0 * value / total:5.1f}%)")
    print(f"sum measured stages: {total:.3f} ms")
    print(f"backward with saved LSE: {backward_saved_lse:.3f} ms")


if __name__ == "__main__":
    main()
