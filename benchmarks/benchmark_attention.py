"""Compare the readable Triton attention kernel with PyTorch SDPA."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.triton.attention import attention_triton


def bench(fn, warmup=10, iters=50) -> float:
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
    for batch, seq_len in ((1, 128), (2, 512)):
        q = torch.randn(batch, 16, seq_len, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, 8, seq_len, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        sdpa_ms = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True))
        triton_ms = bench(lambda: attention_triton(q, k, v))
        print(
            f"B={batch} S={seq_len} Hq=16 Hkv=8 D=128  "
            f"SDPA={sdpa_ms:.3f} ms  Triton={triton_ms:.3f} ms  "
            f"speedup={sdpa_ms / triton_ms:.3f}x"
        )


if __name__ == "__main__":
    main()
