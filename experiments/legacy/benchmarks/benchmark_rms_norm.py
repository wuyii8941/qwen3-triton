from __future__ import annotations

import argparse
import time

import torch

from ops.reference.rms_norm import rms_norm_reference
from ops.triton.rms_norm import rms_norm_triton


CASES = [
    ("hidden_b1_s128", (1, 128, 2048)),
    ("hidden_b1_s512", (1, 512, 2048)),
    ("qk_b1_s128", (1, 128, 8, 128)),
    ("hidden_b4_s512", (4, 512, 2048)),
]


def measure(fn, warmup: int, iterations: int, device: torch.device) -> tuple[float, int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    median = sorted(timings)[len(timings) // 2]
    peak = torch.cuda.max_memory_allocated(device)
    return median, peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"warmup={args.warmup}, iterations={args.iterations}")
    for name, shape in CASES:
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(0)
            x = torch.randn(shape, device=device, dtype=dtype)
            weight = torch.randn(shape[-1], device=device, dtype=dtype)
            reference = rms_norm_reference(x, weight, 1e-6)
            actual = rms_norm_triton(x, weight, 1e-6)
            max_error = (actual.float() - reference.float()).abs().max().item()

            ref_ms, ref_mem = measure(lambda: rms_norm_reference(x, weight, 1e-6), args.warmup, args.iterations, device)
            tri_ms, tri_mem = measure(lambda: rms_norm_triton(x, weight, 1e-6), args.warmup, args.iterations, device)
            print(
                f"{name:18s} {str(dtype):15s} "
                f"ref={ref_ms:8.4f} ms "
                f"triton={tri_ms:8.4f} ms "
                f"speedup={ref_ms / tri_ms:6.3f}x "
                f"error={max_error:.3e} "
                f"mem(ref/triton)={ref_mem / 2**20:.1f}/{tri_mem / 2**20:.1f} MiB"
            )


if __name__ == "__main__":
    main()
