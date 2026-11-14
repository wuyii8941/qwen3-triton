"""Fused Q/K RMSNorm+split-half RoPE with matching baseline BF16 cast edges.

Backward fuses inverse rotation with RMSNorm derivatives. The same deterministic
weight-gradient reduction as the validated unfused baseline is retained.
"""
import torch
import triton
import triton.language as tl
from .norm import _reduce_rows

@triton.jit
def _norm_rope_forward(X,W,C,S,Y,INV,H:tl.constexpr,HEADS:tl.constexpr,EPS:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0);c=tl.arange(0,D);valid=c<H
    paired=tl.where(c<H//2,c+H//2,c-H//2);sign=tl.where(c<H//2,-1.,1.)
    x=tl.load(X+row*H+c,valid,0).to(tl.float32)
    xp=tl.load(X+row*H+paired,valid,0).to(tl.float32)
    w=tl.load(W+c,valid,0).to(tl.float32);wp=tl.load(W+paired,valid,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/H+EPS)
    normal=(x*inv).to(Y.dtype.element_ty).to(tl.float32)
    normalp=(xp*inv).to(Y.dtype.element_ty).to(tl.float32)
    normal=(normal*w).to(Y.dtype.element_ty).to(tl.float32)
    normalp=(normalp*wp).to(Y.dtype.element_ty).to(tl.float32)
    cos=tl.load(C+(row//HEADS)*H+c,valid,0).to(tl.float32)
    sin=tl.load(S+(row//HEADS)*H+c,valid,0).to(tl.float32)
    left=(normal*cos).to(Y.dtype.element_ty).to(tl.float32)
    right=(normalp*sin).to(Y.dtype.element_ty).to(tl.float32)*sign
    tl.store(Y+row*H+c,left+right,valid);tl.store(INV+row,inv)

@triton.jit
def _norm_rope_backward(X,W,C,S,G,INV,DX,P,M:tl.constexpr,H:tl.constexpr,HEADS:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0);r=row+tl.arange(0,1);c=tl.arange(0,D)
    valid=(r[:,None]<M)&(c[None,:]<H)
    paired=tl.where(c<H//2,c+H//2,c-H//2);sign=tl.where(c<H//2,1.,-1.)
    x=tl.load(X+r[:,None]*H+c[None,:],valid,0).to(tl.float32)
    dy=tl.load(G+r[:,None]*H+c[None,:],valid,0).to(tl.float32)
    dyp=tl.load(G+r[:,None]*H+paired[None,:],valid,0).to(tl.float32)
    cos=tl.load(C+(r[:,None]//HEADS)*H+c[None,:],valid,0).to(tl.float32)
    sinp=tl.load(S+(r[:,None]//HEADS)*H+paired[None,:],valid,0).to(tl.float32)
    left=(dy*cos).to(X.dtype.element_ty).to(tl.float32)
    right=(dyp*sinp).to(X.dtype.element_ty).to(tl.float32)*sign[None,:]
    norm_dy=(left+right).to(X.dtype.element_ty).to(tl.float32)
    w=tl.load(W+c,c<H,0).to(tl.float32);inv=tl.load(INV+r,r<M,0)
    weighted=(norm_dy*w[None,:]).to(X.dtype.element_ty).to(tl.float32)
    grad_inv=tl.sum(weighted*x,1)
    grad_var=(-0.5*grad_inv)*((inv*inv)*inv)
    dx=weighted*inv[:,None]+(grad_var/H)[:,None]*(2.*x)
    normalized=(x*inv[:,None]).to(X.dtype.element_ty).to(tl.float32)
    dw=(norm_dy*normalized).to(W.dtype.element_ty).to(tl.float32)
    tl.store(DX+r[:,None]*H+c[None,:],dx,valid)
    tl.store(P+row*H+c,tl.sum(dw,0),c<H)

class _NormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,cos,sin,eps):
        if x.ndim!=4 or w.shape!=(x.shape[-1],) or cos.shape!=sin.shape or cos.shape!=(x.shape[0],x.shape[1],x.shape[-1]):raise ValueError('Expected [B,S,heads,D], [D] and [B,S,D]')
        if not x.is_cuda or any(t.device!=x.device or t.dtype!=x.dtype or not t.is_contiguous() for t in (x,w,cos,sin)):raise ValueError('Matching contiguous CUDA tensors required')
        if x.dtype not in (torch.float32,torch.bfloat16,torch.float16) or x.shape[-1]%2 or x.numel()<1 or eps<=0:raise ValueError('Floating point, even nonempty D and positive epsilon required')
        if cos.requires_grad or sin.requires_grad:raise ValueError('Fixed rotary buffers only')
        h=x.shape[-1];m=x.numel()//h;heads=x.shape[2];y=torch.empty_like(x);inv=torch.empty(m,device=x.device,dtype=torch.float32)
        _norm_rope_forward[(m,)](x,w,cos,sin,y,inv,h,heads,eps,triton.next_power_of_2(h),enable_fp_fusion=False)
        ctx.save_for_backward(x,w,cos,sin,inv);return y
    @staticmethod
    def backward(ctx,dy):
        x,w,cos,sin,inv=ctx.saved_tensors;dy=dy.contiguous();h=x.shape[-1];m=inv.numel();dx=torch.empty_like(x);partial=torch.empty((m,h),device=x.device,dtype=torch.float32)
        _norm_rope_backward[(m,)](x,w,cos,sin,dy,inv,dx,partial,m,h,x.shape[2],triton.next_power_of_2(h),num_warps=4,enable_fp_fusion=False)
        groups=triton.cdiv(m,128);small=torch.empty((groups,h),device=x.device,dtype=torch.float32);dw=torch.empty_like(w)
        _reduce_rows[(groups,triton.cdiv(h,32))](partial,small,m,h,128,32,enable_fp_fusion=False)
        _reduce_rows[(1,triton.cdiv(h,32))](small,dw,groups,h,triton.next_power_of_2(groups),32,enable_fp_fusion=False)
        return dx,dw,None,None,None

def norm_rope(x,weight,cos,sin,eps=1e-6):return _NormRoPE.apply(x,weight,cos,sin,eps)
