"""Qwen split-half rotary tables/application, explicit Triton compute.

inv_freq is a static model buffer loaded/materialized with configuration, not
computed by a hidden GPU PyTorch operation during model forward.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _tables(POS,FREQ,COS,SIN,T:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<T*D
    p=tl.load(POS+i//D,valid,0).to(tl.float32)
    f=tl.load(FREQ+i%(D//2),valid,0).to(tl.float32)
    angle=p*f
    tl.store(COS+i,libdevice.cos(angle),valid);tl.store(SIN+i,libdevice.sin(angle),valid)

@triton.jit
def _apply(X,C,S,Y,N:tl.constexpr,HEADS:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr,BACKWARD:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    d=i%D;paired=tl.where(d<D//2,d+D//2,d-D//2)
    other=i-d+paired;token=i//(HEADS*D)
    x=tl.load(X+i,valid,0).to(tl.float32);xp=tl.load(X+other,valid,0).to(tl.float32)
    c=tl.load(C+token*D+d,valid,0).to(tl.float32)
    if BACKWARD:
        s=tl.load(S+token*D+paired,valid,0).to(tl.float32)
        sign=tl.where(d<D//2,1.,-1.)
    else:
        s=tl.load(S+token*D+d,valid,0).to(tl.float32)
        sign=tl.where(d<D//2,-1.,1.)
    left=(x*c).to(Y.dtype.element_ty).to(tl.float32)
    right=(xp*s).to(Y.dtype.element_ty).to(tl.float32)*sign
    tl.store(Y+i,left+right,valid)

def rotary_tables(position_ids,inv_freq,dtype):
    if not position_ids.is_cuda or inv_freq.device!=position_ids.device:raise ValueError('Same CUDA device required')
    if position_ids.ndim!=2 or not position_ids.is_contiguous() or position_ids.dtype not in (torch.int32,torch.int64):raise ValueError('Contiguous integer [B,S] position IDs required')
    if inv_freq.dtype!=torch.float32 or inv_freq.ndim!=1 or not inv_freq.is_contiguous():raise ValueError('Contiguous FP32 frequency buffer required')
    if dtype not in (torch.float32,torch.bfloat16,torch.float16):raise TypeError('Floating point output required')
    d=inv_freq.numel()*2
    if position_ids.numel()<1 or d<2:raise ValueError('Nonempty positions/frequencies required')
    cos=torch.empty((*position_ids.shape,d),device=position_ids.device,dtype=dtype);sin=torch.empty_like(cos)
    _tables[(triton.cdiv(cos.numel(),256),)](position_ids,inv_freq,cos,sin,position_ids.numel(),d,256,enable_fp_fusion=False)
    return cos,sin

class _Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,cos,sin):
        if x.ndim!=4 or cos.shape!=sin.shape or cos.shape!=(x.shape[0],x.shape[1],x.shape[3]) or x.shape[-1]%2:
            raise ValueError('Expected [B,S,H,D] and matching [B,S,D], even D')
        if not x.is_cuda or any(t.device!=x.device or t.dtype!=x.dtype or not t.is_contiguous() for t in (x,cos,sin)):
            raise ValueError('Same-device/dtype contiguous CUDA tensors required')
        if cos.requires_grad or sin.requires_grad:raise ValueError('Fixed frequency buffers only')
        y=torch.empty_like(x)
        _apply[(triton.cdiv(x.numel(),256),)](x,cos,sin,y,x.numel(),x.shape[2],x.shape[3],256,False,enable_fp_fusion=False)
        ctx.save_for_backward(cos,sin);ctx.shape=x.shape
        return y
    @staticmethod
    def backward(ctx,dy):
        cos,sin=ctx.saved_tensors;dy=dy.contiguous();dx=torch.empty_like(dy)
        _apply[(triton.cdiv(dy.numel(),256),)](dy,cos,sin,dx,dy.numel(),ctx.shape[2],ctx.shape[3],256,True,enable_fp_fusion=False)
        return dx,None,None

def apply_rotary(x,cos,sin):
    return _Rotary.apply(x,cos,sin)
