"""Baseline profiler for a local Qwen causal language model.

This is intentionally a measurement-only startup: it does not replace any
operator.  It records the tensor interface of every Transformer block so
later custom implementations have a concrete reference to match.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "/data1/tzh/models/Qwen/Qwen3-1.7B"
DEFAULT_TEXT = "The goal of this experiment is to understand every operator in a language model."


def _tensor_summary(value: Any) -> dict[str, Any] | None:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, (tuple, list)):
        for item in value:
            summary = _tensor_summary(item)
            if summary is not None:
                return summary
    return None


def _block_hook(index: int, records: list[dict[str, Any]], seen: set[int]):
    def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        # Benchmark warmups call each block repeatedly; retain the first
        # observation because the tensor interface is unchanged.
        if index in seen:
            return
        seen.add(index)
        records.append(
            {
                "block": index,
                "input": _tensor_summary(inputs),
                "output": _tensor_summary(output),
            }
        )

    return hook


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_forward(model: torch.nn.Module, inputs: dict[str, torch.Tensor], device: torch.device,
                     warmup: int, iterations: int) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(**inputs, use_cache=False)
        _synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        timings_ms: list[float] = []
        for _ in range(iterations):
            _synchronize(device)
            start = time.perf_counter()
            model(**inputs, use_cache=False)
            _synchronize(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    result: dict[str, Any] = {
        "warmup_iterations": warmup,
        "measurement_iterations": iterations,
        "latency_ms": {
            "min": min(timings_ms),
            "median": sorted(timings_ms)[len(timings_ms) // 2],
            "mean": sum(timings_ms) / len(timings_ms),
            "all": timings_ms,
        },
    }
    if device.type == "cuda":
        result["cuda_memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    else:
        result["cuda_memory"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Hugging Face model directory")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", default="artifacts/qwen_baseline.json")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"No config.json found under {model_path}")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading model: {model_path}")
    print(f"Device: {device}; dtype: {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    encoded = tokenizer(args.text, return_tensors="pt", add_special_tokens=True)
    inputs = {name: value.to(device) for name, value in encoded.items()}
    records: list[dict[str, Any]] = []
    seen_blocks: set[int] = set()
    handles = []
    layers = getattr(model, "model", None)
    layers = getattr(layers, "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate model.model.layers; inspect this model before continuing")
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(_block_hook(index, records, seen_blocks)))

    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        # Shape hooks are only for capture, never part of latency measurement.
        for handle in handles:
            handle.remove()
        handles.clear()
        measurement = _measure_forward(model, inputs, device, args.warmup, args.iterations)
    finally:
        for handle in handles:
            handle.remove()

    output = {
        "model": str(model_path),
        "device": str(device),
        "dtype": str(dtype),
        "text": args.text,
        "input_ids_shape": list(inputs["input_ids"].shape),
        "num_layers": len(layers),
        "blocks": records,
        "measurement": measurement,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Input shape: {output['input_ids_shape']}")
    print(f"Recorded blocks: {len(records)}")
    print(f"Median forward: {measurement['latency_ms']['median']:.3f} ms")
    if measurement["cuda_memory"] is not None:
        allocated = measurement["cuda_memory"]["peak_allocated_bytes"] / (1024**3)
        reserved = measurement["cuda_memory"]["peak_reserved_bytes"] / (1024**3)
        print(f"Peak CUDA memory: allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB")
    print(f"Saved report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
