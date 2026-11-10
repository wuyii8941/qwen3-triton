"""Minimal trainable entry point for the Tiny-Qwen baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import TinyQwenConfig, TinyQwenForCausalLM


def make_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    # A deterministic next-token pattern gives a tiny model a learnable smoke
    # test without downloading a dataset. TODO(student): replace this with
    # the CS336 data.py/tokenized shard pipeline in the next stage.
    base = torch.arange(seq_len, device=device).unsqueeze(0) % vocab_size
    return base.repeat(batch_size, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rms-norm-impl", choices=["torch", "triton"], default="torch")
    parser.add_argument("--checkpoint", default="artifacts/tiny_qwen_checkpoint.pt")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    config = TinyQwenConfig(
        max_position_embeddings=max(1024, args.seq_len),
        rms_norm_impl=args.rms_norm_impl,
    )
    model = TinyQwenForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    input_ids = make_batch(args.batch_size, args.seq_len, config.vocab_size, device)

    model.train()
    losses: list[float] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids, labels=input_ids)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f"step={step:04d} loss={losses[-1]:.6f}")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": model.config_dict()}, checkpoint_path)
    report_path = checkpoint_path.with_suffix(".json")
    report_path.write_text(json.dumps({"device": str(device), "losses": losses}, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {checkpoint_path.resolve()}")
    print(f"saved loss report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
