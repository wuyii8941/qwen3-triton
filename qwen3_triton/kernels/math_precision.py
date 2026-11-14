"""Unqualified full8B math controls preserving native FP32 rounding order.

Uses the repository's existing all-Triton PyTorch2.10 row reduction helper.
SiLU formula follows native ActivationSiluKernel.cu, with libdevice exp and
round-to-nearest division. No native model arithmetic is called here.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .hf_reduction import hf_mean_square,supports_hf_reduction
from qwen3_triton.kernels.norm import _RMSNorm
from qwen3_triton.kernels.norm_rope import _NormRoPE

@triton.jit
def _hf_norm_forward(X,W,Y,INV,H:tl.constexpr,EPS:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0);c=tl.arange(0,D)
    x=tl.load(X+row*H+c,c<H,0).to(tl.float32);w=tl.load(W+c,c<H,0).to(tl.float32)
    inv=tl.rsqrt(hf_mean_square(X,row*H,H)+EPS)
    normalized=(x*inv).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+row*H+c,normalized*w,c<H);tl.store(INV+row,inv)

class _HFNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,eps):
        assert supports_hf_reduction(x) and x.device==w.device and x.dtype==w.dtype
        assert w.is_contiguous() and w.ndim==1 and w.numel()==x.shape[-1]
        y=torch.empty_like(x);inv=torch.empty(x.numel()//w.numel(),device=x.device,dtype=torch.float32)
        _hf_norm_forward[(inv.numel(),)](x,w,y,inv,w.numel(),eps,triton.next_power_of_2(w.numel()),enable_fp_fusion=False)
        ctx.save_for_backward(x,w,inv);ctx.optimized=False
        return y
    @staticmethod
    def backward(ctx,dy):
        dx,dw,_,_=_RMSNorm.backward(ctx,dy)
        return dx,dw,None

def rms_norm(x,w,eps=1e-6):return _HFNorm.apply(x,w,eps)

@triton.jit
def _hf_norm_rope_forward(X,W,C,S,Y,INV,H:tl.constexpr,HEADS:tl.constexpr,EPS:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0);c=tl.arange(0,D);valid=c<H
    paired=tl.where(c<H//2,c+H//2,c-H//2);sign=tl.where(c<H//2,-1.,1.)
    x=tl.load(X+row*H+c,valid,0).to(tl.float32);xp=tl.load(X+row*H+paired,valid,0).to(tl.float32)
    w=tl.load(W+c,valid,0).to(tl.float32);wp=tl.load(W+paired,valid,0).to(tl.float32)
    inv=tl.rsqrt(hf_mean_square(X,row*H,H)+EPS)
    normal=(x*inv).to(Y.dtype.element_ty).to(tl.float32);normalp=(xp*inv).to(Y.dtype.element_ty).to(tl.float32)
    normal=(normal*w).to(Y.dtype.element_ty).to(tl.float32);normalp=(normalp*wp).to(Y.dtype.element_ty).to(tl.float32)
    cos=tl.load(C+(row//HEADS)*H+c,valid,0).to(tl.float32);sin=tl.load(S+(row//HEADS)*H+c,valid,0).to(tl.float32)
    left=(normal*cos).to(Y.dtype.element_ty).to(tl.float32);right=(normalp*sin).to(Y.dtype.element_ty).to(tl.float32)*sign
    tl.store(Y+row*H+c,left+right,valid);tl.store(INV+row,inv)

class _HFNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,cos,sin,eps):
        assert supports_hf_reduction(x) and x.ndim==4 and w.shape==(x.shape[-1],)
        assert cos.shape==sin.shape==(x.shape[0],x.shape[1],x.shape[-1])
        assert all(t.dtype==x.dtype and t.device==x.device and t.is_contiguous() for t in (x,w,cos,sin))
        h=x.shape[-1];m=x.numel()//h;y=torch.empty_like(x);inv=torch.empty(m,device=x.device,dtype=torch.float32)
        _hf_norm_rope_forward[(m,)](x,w,cos,sin,y,inv,h,x.shape[2],eps,triton.next_power_of_2(h),enable_fp_fusion=False)
        ctx.save_for_backward(x,w,cos,sin,inv);return y
    @staticmethod
    def backward(ctx,dy):return _NormRoPE.backward(ctx,dy)

def norm_rope(x,w,cos,sin,eps=1e-6):return _HFNormRoPE.apply(x,w,cos,sin,eps)

@triton.jit
def _hf_swiglu_forward(G,U,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    g=tl.load(G+i,valid,0).to(tl.float32);u=tl.load(U+i,valid,0).to(tl.float32)
    silu=tl.div_rn(g,1.+libdevice.exp(-g)).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+i,silu*u,valid)

@triton.jit
def _hf_swiglu_backward(G,U,DY,DG,DU,N:tl.constexpr,BLOCK:tl.constexpr,FMA:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    g=tl.load(G+i,valid,0).to(tl.float32);u=tl.load(U+i,valid,0).to(tl.float32);dy=tl.load(DY+i,valid,0).to(tl.float32)
    denom=1.+libdevice.exp(-g);sig=tl.div_rn(1.,denom)
    silu=tl.div_rn(g,denom).to(G.dtype.element_ty).to(tl.float32)
    dact=(dy*u).to(G.dtype.element_ty).to(tl.float32)
    bracket=tl.fma(g,1.-sig,1.) if FMA else 1.+g*(1.-sig)
    tl.store(DG+i,(dact*sig)*bracket,valid);tl.store(DU+i,dy*silu,valid)

class _HFSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,g,u,fma):
        assert g.is_cuda and g.shape==u.shape and g.dtype==u.dtype and g.device==u.device
        assert g.is_contiguous() and u.is_contiguous()
        y=torch.empty_like(g);_hf_swiglu_forward[(triton.cdiv(g.numel(),256),)](g,u,y,g.numel(),256,enable_fp_fusion=False)
        ctx.save_for_backward(g,u);ctx.fma=fma;return y
    @staticmethod
    def backward(ctx,dy):
        g,u=ctx.saved_tensors;dy=dy.contiguous();dg=torch.empty_like(g);du=torch.empty_like(u)
        _hf_swiglu_backward[(triton.cdiv(g.numel(),256),)](g,u,dy,dg,du,g.numel(),256,ctx.fma,enable_fp_fusion=False)
        return dg,du,None

def swiglu(g,u,*,fma=True):return _HFSwiGLU.apply(g,u,fma)
