"""Poison audited outputs to detect missing stores, then compare full gradients.

Uses odd/tail GEMM and SwiGLU dimensions as well as real Qwen decoder shapes.
NaN poisoning is only for these correctness checks, outside all timing. Global
deterministic fill stays True. No fallback kernel is substituted for the model.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
import qwen3_triton.runtime.allocation as allocation
from qwen3_triton.kernels.linear import matmul,linear
from qwen3_triton.kernels.mlp_precision import gate_up_swiglu
from qwen3_triton.validation.audit import NoFallback


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.cuda.set_device(0);torch.manual_seed(20261115)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[];original=allocation.raw_overwrite_output
    def poisoned(shape,*,device,dtype):
        out=original(shape,device=device,dtype=dtype);out.fill_(float('nan'));return out
    for m,n,k in [(17,67,35),(65,129,33),(512,4096,4096)]:
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
        allocation.configure(False)
        with NoFallback():ref=linear(x,w,optimized=True)
        with NoFallback(backward=True):ref.backward(dy)
        expected=[t.detach().clone() for t in [ref,x.grad,w.grad]]
        x.grad=w.grad=None
        allocation.configure(True);allocation.raw_overwrite_output=poisoned
        with NoFallback():got=linear(x,w,optimized=True)
        with NoFallback(backward=True):got.backward(dy)
        checks=[bool(torch.equal(a,b)) and bool(torch.isfinite(a).all()) for a,b in zip([got,x.grad,w.grad],expected)]
        assert all(checks),(m,n,k,checks)
        rows.append(dict(op='linear',shape=[m,n,k],forward_dx_dw_bitwise_equal=checks,poisoned=True))
        allocation.raw_overwrite_output=original
        del x,w,dy,ref,got,expected
    for m,n,k in [(17,67,35),(65,129,33),(512,12288,4096)]:
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)*.01;x=x.detach().requires_grad_()
        wg=(torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.01).requires_grad_()
        wu=(torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.01).requires_grad_()
        dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
        allocation.configure(False)
        with NoFallback():ref=gate_up_swiglu(x,wg,wu,optimized=True)
        with NoFallback(backward=True):ref.backward(dy)
        expected=[t.detach().clone() for t in [ref,x.grad,wg.grad,wu.grad]]
        x.grad=wg.grad=wu.grad=None
        allocation.configure(True);allocation.raw_overwrite_output=poisoned
        with NoFallback():got=gate_up_swiglu(x,wg,wu,optimized=True)
        with NoFallback(backward=True):got.backward(dy)
        checks=[bool(torch.equal(a,b)) and bool(torch.isfinite(a).all()) for a,b in zip([got,x.grad,wg.grad,wu.grad],expected)]
        assert all(checks),(m,n,k,checks)
        rows.append(dict(op='gate_up_swiglu',shape=[m,n,k],forward_dx_dwg_dwu_bitwise_equal=checks,poisoned=True))
        allocation.raw_overwrite_output=original
        del x,wg,wu,dy,ref,got,expected
    assert torch.utils.deterministic.fill_uninitialized_memory is True
    assert torch.are_deterministic_algorithms_enabled()
    root=Path(__file__).resolve().parents[1]
    files=[Path(__file__),root/'qwen3_triton/runtime/allocation.py',root/'qwen3_triton/kernels/linear.py',root/'qwen3_triton/kernels/mlp_precision.py']
    report=dict(complete=True,full_training_qualified=False,rows=rows,deterministic_fill_still_enabled=True,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},scope=__doc__)
    with args.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,all_output_and_gradient_coordinates_bitwise_equal=True,cases=len(rows))),flush=True)


if __name__=='__main__':main()
