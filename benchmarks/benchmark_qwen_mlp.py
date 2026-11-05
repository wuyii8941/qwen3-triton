"""End-to-end Qwen MLP comparison for separate/packed projections."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import TinyQwenConfig  # noqa: E402
from model.qwen_tiny import QwenMLP  # noqa: E402


def measure(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) / iters


def copy_separate_to_packed(separate: QwenMLP, packed: QwenMLP) -> None:
    with torch.no_grad():
        packed.gate_up_proj.weight[: separate.gate_proj.out_features].copy_(separate.gate_proj.weight)
        packed.gate_up_proj.weight[separate.gate_proj.out_features :].copy_(separate.up_proj.weight)
        packed.down_proj.weight.copy_(separate.down_proj.weight)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(60)
    dtype = torch.bfloat16
    batch, seq_len, hidden, intermediate = 1, 2048, 2048, 6144
    x_base = torch.randn(batch, seq_len, hidden, device="cuda", dtype=dtype)
    grad_out = torch.randn_like(x_base)

    base_cfg = dict(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )
    separate_torch = QwenMLP(TinyQwenConfig(**base_cfg, swiglu_impl="torch", packed_gate_up=False)).to("cuda", dtype=dtype)
    separate_triton = QwenMLP(TinyQwenConfig(**base_cfg, swiglu_impl="triton", packed_gate_up=False)).to("cuda", dtype=dtype)
    packed_torch = QwenMLP(TinyQwenConfig(**base_cfg, swiglu_impl="torch", packed_gate_up=True)).to("cuda", dtype=dtype)
    packed_triton = QwenMLP(TinyQwenConfig(**base_cfg, swiglu_impl="triton", packed_gate_up=True)).to("cuda", dtype=dtype)
    fused_gemm = QwenMLP(TinyQwenConfig(**base_cfg, fused_gate_up=True)).to("cuda", dtype=dtype)
    separate_triton.load_state_dict(separate_torch.state_dict())
    copy_separate_to_packed(separate_torch, packed_torch)
    copy_separate_to_packed(separate_torch, packed_triton)
    fused_gemm.gate_proj.weight.data.copy_(separate_torch.gate_proj.weight.data)
    fused_gemm.up_proj.weight.data.copy_(separate_torch.up_proj.weight.data)
    fused_gemm.down_proj.weight.data.copy_(separate_torch.down_proj.weight.data)

    variants = {
        "separate + eager SwiGLU": separate_torch,
        "separate + Triton SwiGLU": separate_triton,
        "packed + eager SwiGLU": packed_torch,
        "packed + Triton SwiGLU": packed_triton,
        "fused GEMM + SwiGLU": fused_gemm,
    }
    print(f"shape=[{batch}, {seq_len}, {hidden}], intermediate={intermediate}, dtype={dtype}")
    for name, module in variants.items():
        x = x_base.detach().clone().requires_grad_()

        def step(module=module, x=x):
            module.zero_grad(set_to_none=True)
            x.grad = None
            module(x).backward(grad_out)

        ms = measure(step)
        torch.cuda.reset_peak_memory_stats()
        step(); torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"{name:28s} {ms:8.3f} ms  peak={peak_mib:8.1f} MiB")


if __name__ == "__main__":
    main()
