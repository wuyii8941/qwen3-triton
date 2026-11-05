"""Real Qwen first-layer weights, synthetic decode activations; not E2E."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
from transformers import AutoModelForCausalLM
from model.hf_inference import _activation
from ops.triton.decode_gate_up import decode_gate_up_triton
from benchmark_inference_evidence import timing
from benchmark_hf_inference import check


def activation(g,u):
    y=torch.empty_like(g)
    _activation[(triton.cdiv(y.numel(),512),)](g,u,y,y.numel(),512,enable_fp_fusion=False)
    return y


@torch.inference_mode()
def main():
    torch.manual_seed(2026)
    model=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-1.7B',
        local_files_only=True,torch_dtype=torch.float32)
    gw=model.model.layers[0].mlp.gate_proj.weight.detach().clone()
    uw=model.model.layers[0].mlp.up_proj.weight.detach().clone()
    del model
    rows=[]
    for dtype in (torch.float16,torch.bfloat16):
        g,u=gw.cuda().to(dtype),uw.cuda().to(dtype)
        packed=torch.cat([g,u],0)
        for m in (1,2,4,8,16):
            x=torch.randn(m,2048,device='cuda',dtype=dtype)
            expected=F.silu(F.linear(x,g))*F.linear(x,u)
            def separate():
                return activation(F.linear(x,g),F.linear(x,u))
            def packed_fn():
                z=F.linear(x,packed)
                # Views have a nontrivial row stride: make contiguous for the
                # existing flat activation kernel; include these copies in timing.
                a,b=z.chunk(2,dim=-1)
                return activation(a.contiguous(),b.contiguous())
            variants={'two_cublas+triton_activation':separate,'packed_cublas+copies':packed_fn}
            for bn in (1,2,4,8):
                variants[f'gemv_bn{bn}']=lambda bn=bn:decode_gate_up_triton(x,g,u,block_n=bn)
            for name,fn in variants.items():
                actual=fn()
                error=check(actual,expected)
                row=dict(dtype=str(dtype),m=m,backend=name,error=error,**timing(fn))
                rows.append(row)
                print(dtype,m,name,round(row['gpu_graph_median_ms']*1000,2),error,flush=True)
    root=Path(__file__).resolve().parents[1]
    (root/'artifacts/decode_gate_up_ablation.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    main()
