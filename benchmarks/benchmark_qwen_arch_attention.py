"""Forward benchmark with Qwen3-1.7B attention dimensions."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import TinyQwenConfig, TinyQwenForCausalLM


def bench(fn, warmup: int = 5, iters: int = 20) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**30


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = TinyQwenConfig(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        rms_norm_impl="torch",
        qk_norm_rope_impl="fused_triton",
    )
    input_ids = torch.randint(config.vocab_size, (1, 128), device=device)

    models = {}
    for name, attention_impl in (("sdpa", "sdpa"), ("triton_attention", "triton")):
        cfg = TinyQwenConfig(**config.__dict__)
        cfg.attention_impl = attention_impl
        models[name] = TinyQwenForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    models["triton_attention"].load_state_dict(models["sdpa"].state_dict())

    with torch.inference_mode():
        reference_hidden = models["sdpa"].model(input_ids)
        actual_hidden = models["triton_attention"].model(input_ids)
        # The hidden state is the tighter structural parity check.  BF16
        # attention accumulation can be amplified by the large LM head.
        torch.testing.assert_close(actual_hidden, reference_hidden, rtol=8e-2, atol=8e-2)
        reference = models["sdpa"](input_ids)["logits"]
        actual = models["triton_attention"](input_ids)["logits"]
        torch.testing.assert_close(actual, reference, rtol=1e-1, atol=5e-1)

    results = {}
    with torch.inference_mode():
        for name, model in models.items():
            latency, memory = bench(lambda: model(input_ids))
            results[name] = latency, memory

    for name, (latency, memory) in results.items():
        print(f"{name:18s} forward={latency:8.3f} ms  peak={memory:.3f} GiB")
    print(f"triton/sdpa latency ratio: {results['triton_attention'][0] / results['sdpa'][0]:.3f}x")


if __name__ == "__main__":
    main()
