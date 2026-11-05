"""Inference Add+RMSNorm: FlashInfer-inspired fusion, HF rounding semantics.

Returns both the updated residual and normalized activation without modifying
inputs; the residual must survive for the subsequent MLP skip connection.
"""
import torch
import triton
import triton.language as tl
from ops import compile_ops


@triton.jit
def _add_norm(X, R, W, S, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X+row*N+d, d < N, 0).to(tl.float32)
    r = tl.load(R+row*N+d, d < N, 0).to(tl.float32)
    w = tl.load(W+d, d < N, 0).to(tl.float32)
    # HF stores the residual sum in activation dtype before normalization.
    summed = (x+r).to(S.dtype.element_ty)
    s = summed.to(tl.float32)
    z = s*tl.rsqrt(tl.sum(s*s, 0)/N+EPS)
    z = z.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(S+row*N+d, summed, d < N)
    tl.store(Y+row*N+d, z*w, d < N)


def add_rms_norm_triton(x, residual, weight, eps=1e-6):
    if torch.compiler.is_compiling():
        return compile_ops.add_norm(x, residual, weight, eps)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x,residual,weight)):
        raise ValueError('Add+RMSNorm is inference-only')
    if x.shape != residual.shape or x.ndim < 1 or weight.shape != (x.shape[-1],):
        raise ValueError('Expected matching activations and weight [hidden]')
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('Unsupported dtype')
    if not x.is_cuda or any(t.device != x.device or t.dtype != x.dtype for t in (residual,weight)):
        raise ValueError('Inputs must share CUDA device and dtype')
    x,residual,weight = x.contiguous(),residual.contiguous(),weight.contiguous()
    summed, normalized = torch.empty_like(x), torch.empty_like(x)
    n=x.shape[-1]
    if n <= 0 or n > 8192:
        raise ValueError('hidden dimension must be 1..8192')
    if x.numel():
        _add_norm[(x.numel()//n,)](x,residual,weight,summed,normalized,n,eps,
                                 triton.next_power_of_2(n),enable_fp_fusion=False)
    return summed,normalized
