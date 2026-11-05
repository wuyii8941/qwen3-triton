"""Reproducible inference evidence; no silent compile fallback or training timing."""
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.reference.rms_norm import rms_norm_reference
from ops.reference.rope import apply_rope_reference
from ops.reference.swiglu import swiglu_reference
from ops.triton.rms_norm import rms_norm_triton
from ops.triton.rope import apply_rope_triton
from ops.triton.silu_and_mul import silu_and_mul_triton
from model.config import TinyQwenConfig
from model.qwen_tiny import QwenDecoderBlock, RotaryEmbedding


def outputs(y):
    return y if isinstance(y, tuple) else (y,)


def timing(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    # Capture identical repeated workloads: GPU timing excludes Python dispatch
    # and JIT. Host-visible timing below measures uncaptured invocation overhead.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            retained = fn()
    samples = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) / 10)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000 / 30
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    delta = torch.cuda.max_memory_allocated() - base
    return dict(gpu_graph_median_ms=statistics.median(samples),
                gpu_graph_min_ms=min(samples), gpu_graph_max_ms=max(samples),
                wall_ms=wall_ms, peak_allocated_delta_bytes=delta)


def compare(label, reference, candidate, args, metadata, compile_reference=True):
    expected = reference(*args)
    variants = {"eager": reference, "triton": candidate}
    if compile_reference:
        # Reset per case to avoid Dynamo's per-code-object recompile limit.
        torch.compiler.reset()
        variants["compile"] = torch.compile(reference, fullgraph=True, dynamic=False,
                                             options={"triton.cudagraphs": False})
    rows = []
    for name, fn in variants.items():
        start = time.perf_counter()
        actual = fn(*args)
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - start) * 1000
        errors = []
        for a, e in zip(outputs(actual), outputs(expected)):
            torch.testing.assert_close(a, e, rtol=0.02, atol=0.02)
            errors.append(float((a.float() - e.float()).abs().max()))
        row = dict(case=label, backend=name, **metadata, max_abs_error=max(errors),
                   first_call_ms=first_ms, **timing(lambda: fn(*args)))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


@torch.inference_mode()
def main():
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for batch, seq in ((1, 1), (1, 128), (2, 512)):
            meta = dict(dtype=str(dtype), batch=batch, seq_len=seq)
            x = torch.randn(batch, seq, 2048, device="cuda", dtype=dtype)
            w = torch.ones(2048, device="cuda", dtype=dtype)
            rows += compare("rmsnorm", rms_norm_reference, rms_norm_triton, (x, w, 1e-6), meta)
            gate = torch.randn(batch, seq, 6144, device="cuda", dtype=dtype)
            up = torch.randn_like(gate)
            rows += compare("swiglu", swiglu_reference, silu_and_mul_triton, (gate, up), meta)
            q = torch.randn(batch, 16, seq, 128, device="cuda", dtype=dtype)
            k = torch.randn(batch, 8, seq, 128, device="cuda", dtype=dtype)
            theta = torch.randn(batch, seq, 64, device="cuda")
            theta = torch.cat((theta, theta), dim=-1)
            rows += compare("rope", apply_rope_reference, apply_rope_triton,
                            (q, k, theta.cos().to(dtype), theta.sin().to(dtype)), meta)
        cfg = TinyQwenConfig(hidden_size=2048, intermediate_size=6144,
                             num_attention_heads=16, num_key_value_heads=8, head_dim=128)
        base = QwenDecoderBlock(cfg).cuda().to(dtype).eval()
        cfg.rms_norm_impl = "triton"
        cfg.swiglu_impl = "triton"
        cfg.qk_norm_rope_impl = "fused_triton"
        fused = QwenDecoderBlock(cfg).cuda().to(dtype).eval()
        fused.load_state_dict(base.state_dict())
        rotary = RotaryEmbedding(cfg).cuda()
        for batch, seq in ((1, 128), (2, 512)):
            x = torch.randn(batch, seq, 2048, device="cuda", dtype=dtype)
            positions = torch.arange(seq, device="cuda").expand(batch, -1)
            cos_sin = rotary(x, positions)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                rows += compare("qwen_shaped_block", base, fused, (x, cos_sin),
                                dict(dtype=str(dtype), batch=batch, seq_len=seq), False)
        del base, fused
    report = dict(manifest=dict(gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                  seed=2026, tolerance="rtol=0.02, atol=0.02",
                  scope="inference operators and randomly initialized Qwen-shaped block; no KV cache"), rows=rows)
    path = Path(__file__).resolve().parents[1] / "artifacts/inference_evidence.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
