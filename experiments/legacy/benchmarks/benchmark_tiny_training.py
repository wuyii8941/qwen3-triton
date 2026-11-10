from __future__ import annotations

import argparse
import time

import torch

from model import TinyQwenConfig, TinyQwenForCausalLM


def train_step(model, optimizer, input_ids, device):
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids, labels=input_ids)["loss"]
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    return loss.detach()


def measure(model, optimizer, input_ids, device, warmup: int, iterations: int):
    for _ in range(warmup):
        train_step(model, optimizer, input_ids, device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    losses = []
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss = train_step(model, optimizer, input_ids, device)
        timings.append((time.perf_counter() - start) * 1000.0)
        losses.append(float(loss.cpu()))
    return sorted(timings)[len(timings) // 2], max(losses), torch.cuda.max_memory_allocated(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    config = TinyQwenConfig(rms_norm_impl="torch")
    torch_model = TinyQwenForCausalLM(config).to(device).train()
    triton_config = TinyQwenConfig(rms_norm_impl="triton")
    triton_model = TinyQwenForCausalLM(triton_config).to(device).train()
    triton_model.load_state_dict(torch_model.state_dict())
    torch_optimizer = torch.optim.AdamW(torch_model.parameters(), lr=3e-4)
    triton_optimizer = torch.optim.AdamW(triton_model.parameters(), lr=3e-4)
    input_ids = torch.randint(config.vocab_size, (args.batch_size, args.seq_len), device=device)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"batch={args.batch_size}, seq_len={args.seq_len}, warmup={args.warmup}, iterations={args.iterations}")
    for name, model, optimizer in (
        ("torch", torch_model, torch_optimizer),
        ("triton_rmsnorm", triton_model, triton_optimizer),
    ):
        median_ms, max_loss, peak_memory = measure(
            model, optimizer, input_ids, device, args.warmup, args.iterations
        )
        print(
            f"{name:16s} step={median_ms:8.3f} ms "
            f"tokens/s={args.batch_size * args.seq_len / (median_ms / 1000):10.1f} "
            f"peak_memory={peak_memory / 2**30:.3f} GiB "
            f"max_loss={max_loss:.5f}"
        )


if __name__ == "__main__":
    main()
