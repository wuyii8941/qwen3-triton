"""Compare PyTorch SDPA and trainable Triton attention."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.triton.attention import attention_triton_autograd


def measure(fn, warmup=5, iters=20) -> float:
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
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(0)
        q = torch.randn(1, 16, 128, 128, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(1, 8, 128, 128, device="cuda", dtype=dtype, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        grad_out = torch.randn_like(q)

        def sdpa_step() -> None:
            q.grad = k.grad = v.grad = None
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            out.backward(grad_out)

        def triton_step() -> None:
            q.grad = k.grad = v.grad = None
            out = attention_triton_autograd(q, k, v)
            out.backward(grad_out)

        sdpa_ms = measure(sdpa_step)
        triton_ms = measure(triton_step)
        print(
            f"dtype={dtype} B=1 Hq=16 Hkv=8 S=128 D=128 "
            f"SDPA={sdpa_ms:.3f} ms Triton={triton_ms:.3f} ms "
            f"Triton/SDPA={triton_ms / sdpa_ms:.3f}x"
        )


if __name__ == "__main__":
    main()
