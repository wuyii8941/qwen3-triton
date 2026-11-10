"""Experimental small-M gate/up GEMV + SwiGLU, preserving HF casts.

One program owns a token and a small group of output features. Reduction spans
the whole hidden dimension, avoiding Tensor Core padding for M=1. This trades
weight reuse across tokens for greater output-channel parallelism: benchmark
before selecting it. This is not a reproduction of DeepFusionKernel.
"""
import torch
import triton
import triton.language as tl
from ops import compile_ops


@triton.jit
def _decode_gate_up(X,G,U,Y,N:tl.constexpr,K:tl.constexpr,
                    BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    x=tl.load(X+m*K+k,k<K,0).to(tl.float32)
    g=tl.load(G+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
    u=tl.load(U+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
    gate=tl.sum(g*x[None,:],1).to(Y.dtype.element_ty).to(tl.float32)
    up=tl.sum(u*x[None,:],1).to(Y.dtype.element_ty).to(tl.float32)
    silu=(gate/(1+tl.exp(-gate))).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+m*N+n,silu*up,n<N)


def decode_gate_up_triton(x,gate_weight,up_weight,*,block_n=4,num_warps=4):
    if torch.compiler.is_compiling():
        return compile_ops.decode_gate_up(x, gate_weight, up_weight, block_n, num_warps)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x,gate_weight,up_weight)):
        raise ValueError('Inference only')
    if x.ndim!=2 or gate_weight.ndim!=2 or up_weight.shape!=gate_weight.shape:
        raise ValueError('Expected x [M,K] and matching weights [N,K]')
    m,k=x.shape
    n=gate_weight.shape[0]
    if not 1<=m<=16 or not 1<=k<=4096 or gate_weight.shape[1]!=k or n<1:
        raise ValueError('Supported M=1..16, K=1..4096, N>0')
    if x.dtype not in (torch.float16,torch.bfloat16) or not x.is_cuda:
        raise ValueError('CUDA FP16/BF16 required')
    if any(t.device!=x.device or t.dtype!=x.dtype or not t.is_contiguous()
           for t in (x,gate_weight,up_weight)):
        raise ValueError('Contiguous same-device/same-dtype tensors required')
    if block_n not in (1,2,4,8) or num_warps not in (4,8):
        raise ValueError('Unsupported launch configuration')
    y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    _decode_gate_up[(m,triton.cdiv(n,block_n))](x,gate_weight,up_weight,y,n,k,
        block_n,triton.next_power_of_2(k),num_warps=num_warps,enable_fp_fusion=False)
    return y
