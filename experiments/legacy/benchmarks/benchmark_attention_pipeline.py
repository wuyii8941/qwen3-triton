"""Compare compact Triton prefill/decode with native PyTorch SDPA."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.triton.attention_pipeline import (  # noqa: E402
    CompactGQAAttentionPipeline,
    ContiguousKVCache,
    sdpa_decode,
    sdpa_prefill,
)


def measure(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    b, hq, hkv, d = 1, 16, 8, 128
    pipeline = CompactGQAAttentionPipeline(hq, hkv, d)
    for seq_len in (128, 512, 2048):
        q = torch.randn(b, hq, seq_len, d, device="cuda", dtype=dtype)
        k = torch.randn(b, hkv, seq_len, d, device="cuda", dtype=dtype)
        v = torch.randn_like(k)

        def triton_prefill():
            cache = ContiguousKVCache(b, hkv, seq_len, d, device=q.device, dtype=dtype)
            pipeline.prefill(q, k, v, cache)

        def flash_prefill():
            sdpa_prefill(q, k, v)

        t = measure(triton_prefill)
        f = measure(flash_prefill)
        print(f"prefill S={seq_len:4d}: Triton={t:.3f} ms SDPA={f:.3f} ms ratio={t/f:.3f}x")

    for cache_len in (128, 512, 2048, 4096):
        q = torch.randn(b, hq, 1, d, device="cuda", dtype=dtype)
        k_cache = torch.randn(b, hkv, cache_len, d, device="cuda", dtype=dtype)
        v_cache = torch.randn_like(k_cache)

        def triton_decode():
            from ops.triton.attention import attention_triton_decode
            attention_triton_decode(q, k_cache, v_cache)

        def flash_decode():
            sdpa_decode(q, k_cache, v_cache)

        t = measure(triton_decode)
        f = measure(flash_decode)
        print(f"decode  S={cache_len:4d}: Triton={t:.3f} ms SDPA={f:.3f} ms ratio={t/f:.3f}x")


if __name__ == "__main__":
    main()
