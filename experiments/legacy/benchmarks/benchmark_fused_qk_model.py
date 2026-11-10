"""End-to-end Tiny-Qwen benchmark for the Q/K Norm + RoPE fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow both ``python benchmarks/...py`` and ``python -m benchmarks...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import TinyQwenConfig, TinyQwenForCausalLM


def timed(fn, warmup: int = 10, iters: int = 50) -> tuple[float, float]:
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
    return start.elapsed_time(end) / iters, float(torch.cuda.max_memory_allocated()) / 2**30


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cases = [
        ("smoke", 256, 768, 2, 4, 2, 4, 128),
        ("qwen3_arch", 512, 1536, 2, 4, 2, 2, 128),
        ("qwen3_1layer", 2048, 6144, 1, 16, 8, 1, 128),
    ]
    for label, hidden, intermediate, layers, heads, kv_heads, batch, seq_len in cases:
        config = TinyQwenConfig(
            vocab_size=4096,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=hidden // heads,
            max_position_embeddings=256,
            rms_norm_impl="torch",
        )
        input_ids = torch.randint(config.vocab_size, (batch, seq_len), device=device)
        models = {}
        for name, impl in (("torch", "torch"), ("fused_qk_norm_rope", "fused_triton")):
            cfg = TinyQwenConfig(**config.__dict__)
            cfg.qk_norm_rope_impl = impl
            models[name] = TinyQwenForCausalLM(cfg).to(device=device, dtype=dtype)
        models["fused_qk_norm_rope"].load_state_dict(models["torch"].state_dict())

        results = {}
        for name, model in models.items():
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                forward_ms, forward_mem = timed(lambda: model(input_ids), warmup=5, iters=20)

            model.train()

            def train_step() -> None:
                model.zero_grad(set_to_none=True)
                result = model(input_ids, labels=input_ids)
                result["loss"].backward()

            torch.cuda.reset_peak_memory_stats()
            train_ms, train_mem = timed(train_step, warmup=3, iters=10)
            results[name] = (forward_ms, forward_mem, train_ms, train_mem)

        torch_result = results["torch"]
        fused_result = results["fused_qk_norm_rope"]
        print(
            f"{label:12s} shape=[B={batch},S={seq_len},H={hidden}] "
            f"D={hidden // heads} layers={layers}"
        )
        print(
            f"  torch  forward={torch_result[0]:8.3f} ms  train={torch_result[2]:8.3f} ms  "
            f"mem={torch_result[1]:.3f}/{torch_result[3]:.3f} GiB"
        )
        print(
            f"  fused  forward={fused_result[0]:8.3f} ms  train={fused_result[2]:8.3f} ms  "
            f"mem={fused_result[1]:.3f}/{fused_result[3]:.3f} GiB  "
            f"speedup={torch_result[0] / fused_result[0]:.3f}x/"
            f"{torch_result[2] / fused_result[2]:.3f}x"
        )


if __name__ == "__main__":
    main()
