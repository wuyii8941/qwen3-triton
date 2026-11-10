"""Operator ablation: HF eager, existing Triton norm + add, fused, compiled."""
import json
from pathlib import Path
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from model.hf_inference import _norm_forward
from ops.triton.add_rms_norm import add_rms_norm_triton
from benchmark_inference_evidence import timing


@torch.inference_mode()
def main():
    torch.manual_seed(2026)
    rows = []
    for dtype in [torch.float16, torch.bfloat16]:
        norm = Qwen3RMSNorm(2048, eps=1e-6).cuda().to(dtype)
        for b,s in [(1,1),(2,1),(1,128),(2,512)]:
            x = torch.randn(b,s,2048,device='cuda',dtype=dtype)
            r = torch.randn_like(x)
            def eager(x,r):
                z = x+r
                return z,norm(z)
            def separate(x,r):
                z = x+r
                return z,_norm_forward(norm,z)
            torch.compiler.reset()
            compiled = torch.compile(eager,fullgraph=True,dynamic=False)
            variants = {'eager':eager, 'separate_triton':separate,
                        'fused':lambda x,r:add_rms_norm_triton(x,r,norm.weight,1e-6),
                        'compile':compiled}
            expected = eager(x,r)
            for name,fn in variants.items():
                actual = fn(x,r)
                torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)
                torch.testing.assert_close(actual[1],expected[1],atol=.002,rtol=.02)
                row = dict(dtype=str(dtype),batch=b,seq_len=s,backend=name,
                           **timing(lambda:fn(x,r)))
                rows.append(row)
                print(row,flush=True)
    target=Path(__file__).resolve().parents[1]/'artifacts/add_rms_norm_ablation.json'
    target.write_text(json.dumps(rows,indent=2)+'\n')


if __name__ == '__main__':
    main()
