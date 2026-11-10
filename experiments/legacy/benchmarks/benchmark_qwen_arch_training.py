"""Compare one-layer Qwen3-shaped training steps."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import TinyQwenConfig, TinyQwenForCausalLM


def measure(fn, warmup=3, iters=10) -> tuple[float, float]:
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
        qk_norm_rope_impl="torch",
    )
    input_ids = torch.randint(config.vocab_size, (1, 128), device="cuda")
    models = {}
    for name, impl in (("sdpa", "sdpa"), ("triton", "triton")):
        cfg = TinyQwenConfig(**config.__dict__)
        cfg.attention_impl = impl
        models[name] = TinyQwenForCausalLM(cfg).cuda().to(dtype=torch.bfloat16)
    models["triton"].load_state_dict(models["sdpa"].state_dict())

    for name, model in models.items():
        model.train()

        def step() -> None:
            model.zero_grad(set_to_none=True)
            result = model(input_ids, labels=input_ids)
            result["loss"].backward()

        latency, memory = measure(step)
        print(f"{name:8s} train_step={latency:.3f} ms peak={memory:.3f} GiB")


if __name__ == "__main__":
    main()
