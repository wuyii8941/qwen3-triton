"""Isolate transpose-copy removal, using identical kernel arithmetic."""
import json
from pathlib import Path
import torch
import triton
from benchmark_inference_evidence import timing
from ops.triton.fused_qk_norm_rope import _fused_one, _fused_qk_norm_rope_kernel


def old_layout(x, w, cos, sin):
    b, s, h, d = x.shape
    source = x.transpose(1, 2).contiguous()
    target = torch.empty_like(source)
    _fused_qk_norm_rope_kernel[(b*h*s,)](
        source, target, cos, sin, w, h, s, d, 1e-6,
        BLOCK_SIZE=triton.next_power_of_2(d), BSHD=False)
    return target.transpose(1, 2).contiguous()


@torch.inference_mode()
def main():
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for b, s in ((1, 128), (2, 512)):
            x = torch.randn(b, s, 16, 128, device='cuda', dtype=dtype)
            w = torch.randn(128, device='cuda', dtype=dtype)
            angle = torch.randn(b, s, 64, device='cuda').repeat(1, 1, 2)
            cos, sin = angle.cos().to(dtype), angle.sin().to(dtype)
            old = lambda: old_layout(x, w, cos, sin)
            new = lambda: _fused_one(x, w, cos, sin, 1e-6)
            torch.testing.assert_close(old(), new(), rtol=0, atol=0)
            row = dict(dtype=str(dtype), batch=b, seq_len=s, before=timing(old), after=timing(new))
            rows.append(row)
            print(json.dumps(row), flush=True)
    (Path(__file__).resolve().parents[1]/'artifacts/qk_layout_ablation.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
