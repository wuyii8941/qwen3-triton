"""Compare packed PyTorch GEMM+SwiGLU with fused Triton GEMM+SwiGLU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.triton.gate_up_swiglu import gate_up_swiglu_triton  # noqa: E402


def measure(fn, warmup=10, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters): fn()
    b.record(); b.synchronize()
    return a.elapsed_time(b) / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    for dtype in (torch.float32, torch.bfloat16):
        for m in (128, 512, 2048):
            k, n = 2048, 6144
            x = torch.randn(m, k, device="cuda", dtype=dtype)
            gate_w = torch.randn(n, k, device="cuda", dtype=dtype)
            up_w = torch.randn_like(gate_w)
            packed_w = torch.cat((gate_w, up_w), dim=0)

            def packed():
                gate_up = F.linear(x, packed_w)
                gate, up = gate_up.chunk(2, dim=-1)
                return F.silu(gate) * up

            def fused():
                return gate_up_swiglu_triton(x, gate_w, up_w)

            packed_ms = measure(packed)
            fused_ms = measure(fused)
            print(
                f"dtype={dtype} M={m} K={k} N={n} "
                f"packed={packed_ms:.3f} ms fused={fused_ms:.3f} ms "
                f"ratio={fused_ms / packed_ms:.3f}x"
            )


if __name__ == "__main__":
    main()
