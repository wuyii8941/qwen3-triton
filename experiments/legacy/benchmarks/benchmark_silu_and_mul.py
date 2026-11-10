"""Benchmark eager PyTorch versus the Triton SwiGLU activation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.triton.silu_and_mul import silu_and_mul_triton  # noqa: E402


def measure(fn, warmup=20, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters): fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    for dtype in (torch.float32, torch.bfloat16):
        for m, n in ((128, 768), (512, 3072), (2048, 6144)):
            # Each shape/dtype has inference and autograd specializations.
            # Reset per case rather than silently hitting Dynamo's limit.
            torch.compiler.reset()
            gate = torch.randn(m, n, device="cuda", dtype=dtype)
            up = torch.randn_like(gate)
            grad_out = torch.randn_like(gate)

            # Forward-only comparison: Inductor is allowed to fuse the eager
            # expression, while Triton uses one explicit kernel.
            eager_forward = lambda: F.silu(gate) * up
            compiled_forward_fn = torch.compile(
                lambda x, y: F.silu(x) * y,
                mode="max-autotune",
                dynamic=False,
                fullgraph=True,
            )
            compiled_forward = lambda: compiled_forward_fn(gate, up)
            triton_forward = lambda: silu_and_mul_triton(gate, up)
            eager_fwd_ms = measure(eager_forward)
            compiled_fwd_ms = measure(compiled_forward)
            triton_fwd_ms = measure(triton_forward)

            # Training comparison: all three paths execute forward and
            # backward with the same upstream gradient.  The Triton custom
            # autograd function uses Triton for both kernels.
            eager_gate = gate.detach().clone().requires_grad_()
            eager_up = up.detach().clone().requires_grad_()
            compiled_gate = gate.detach().clone().requires_grad_()
            compiled_up = up.detach().clone().requires_grad_()
            triton_gate = gate.detach().clone().requires_grad_()
            triton_up = up.detach().clone().requires_grad_()

            def eager_train():
                eager_gate.grad = eager_up.grad = None
                (F.silu(eager_gate) * eager_up).backward(grad_out)

            def compiled_train():
                compiled_gate.grad = compiled_up.grad = None
                compiled_forward_fn(compiled_gate, compiled_up).backward(grad_out)

            def triton_train():
                triton_gate.grad = triton_up.grad = None
                silu_and_mul_triton(triton_gate, triton_up).backward(grad_out)

            eager_ms = measure(eager_train)
            compiled_ms = measure(compiled_train)
            triton_ms = measure(triton_train)
            print(
                f"dtype={dtype} shape=({m},{n}) "
                f"fwd[eager={eager_fwd_ms:.3f},compile={compiled_fwd_ms:.3f},triton={triton_fwd_ms:.3f}] ms "
                f"train[eager={eager_ms:.3f},compile={compiled_ms:.3f},triton={triton_ms:.3f}] ms"
            )


if __name__ == "__main__":
    main()
