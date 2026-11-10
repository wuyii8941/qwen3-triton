"""Validate whether fused dK/dV atomic contention hurts performance."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.triton.attention_backward import attention_kv_backward_triton


def timed(fn, warmup=3, iters=10) -> float:
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
    dtype = torch.bfloat16
    for seq_len in (64, 128, 512):
        for q_heads, kv_heads in ((8, 8), (16, 8), (16, 4), (16, 1)):
            torch.manual_seed(seq_len + q_heads + kv_heads)
            q = torch.randn(1, q_heads, seq_len, 128, device="cuda", dtype=dtype)
            k = torch.randn(1, kv_heads, seq_len, 128, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            grad_out = torch.randn_like(q)
            separate = timed(lambda: attention_kv_backward_triton(q, k, v, grad_out, fused=False))
            fused = timed(lambda: attention_kv_backward_triton(q, k, v, grad_out, fused=True))
            print(
                f"S={seq_len:4d} Hq/Hkv={q_heads:2d}/{kv_heads:2d} "
                f"separate={separate:8.3f} ms fused={fused:8.3f} ms "
                f"fused/separate={fused / separate:.3f}x"
            )


if __name__ == "__main__":
    main()
