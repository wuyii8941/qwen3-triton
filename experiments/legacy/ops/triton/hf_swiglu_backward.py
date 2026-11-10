"""HF-rounded SwiGLU backward: cast grad*silu/up products at native edges."""
import torch
import triton
import triton.language as tl


@triton.jit
def _backward(G,U,D,DG,DU,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    g=tl.load(G+i,i<N,0).to(tl.float32)
    u=tl.load(U+i,i<N,0).to(tl.float32)
    d=tl.load(D+i,i<N,0).to(tl.float32)
    sig=1./(1.+tl.exp(-g))
    silu=(g*sig).to(G.dtype.element_ty).to(tl.float32)
    grad_silu=(d*u).to(G.dtype.element_ty).to(tl.float32)
    dg=grad_silu*sig*(1.+g*(1.-sig))
    du=d*silu
    tl.store(DG+i,dg,i<N);tl.store(DU+i,du,i<N)


def swiglu_backward(g,u,grad):
    g,u,grad=g.contiguous(),u.contiguous(),grad.contiguous()
    dg,du=torch.empty_like(g),torch.empty_like(u)
    _backward[(triton.cdiv(g.numel(),512),)](g,u,grad,dg,du,g.numel(),512,enable_fp_fusion=False)
    return dg,du
