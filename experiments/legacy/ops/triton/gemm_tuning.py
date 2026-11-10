"""Explicit launch experiments on the existing fused GEMM kernel."""
import torch
import triton
from .gate_up_swiglu import _gate_up_swiglu_kernel
from ops import compile_ops

# Resolve the undecorated JIT function outside Dynamo tracing. Accessing an
# autotuner's .fn inside a compiled forward is unsupported in torch 2.5.1.
_gate_up_fixed = _gate_up_swiglu_kernel.fn


def gate_up_tuned(x,g,u,*,bm=32,bn=64,bk=32,warps=4,stages=3,group_m=0,hf_round=True):
    if torch.compiler.is_compiling():
        return compile_ops.gate_up(x, g, u, bm, bn, bk, warps, stages, group_m, hf_round)
    if x.ndim!=2 or g.ndim!=2 or g.shape!=u.shape or x.shape[1]!=g.shape[1]:
        raise ValueError('Expected X [M,K] and weights [N,K]')
    if not x.is_cuda or x.dtype not in (torch.float16,torch.bfloat16):
        raise ValueError('CUDA FP16/BF16 required')
    if any(a.dtype!=x.dtype or a.device!=x.device or not a.is_contiguous() for a in (x,g,u)):
        raise ValueError('Contiguous, matching dtype/device required')
    if torch.is_grad_enabled() and any(a.requires_grad for a in (x,g,u)):
        raise ValueError('Inference only')
    if bm not in (16,32,64,128) or bn not in (32,64,128) or bk not in (32,64,128):
        raise ValueError('Unsupported tile')
    if warps not in (4,8) or stages not in (2,3,4) or group_m not in (0,1,4,8):
        raise ValueError('Unsupported launch configuration')
    m,k=x.shape
    n=g.shape[0]
    if min(m,k,n)<1:
        raise ValueError('Nonempty matrices required')
    y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    grid=(triton.cdiv(m,bm),triton.cdiv(n,bn)) if group_m==0 else (triton.cdiv(m,bm)*triton.cdiv(n,bn),)
    # .fn bypasses autotune so every candidate can be independently reported.
    _gate_up_fixed[grid](x,g,u,y,m,n,k,*x.stride(),*g.stride(),*y.stride(),
        BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,MAX_K=k,GROUP_M=group_m,HF_ROUND=hf_round,
        num_warps=warps,num_stages=stages)
    return y
