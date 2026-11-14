"""RMSNorm with HF BF16 cast boundaries and deterministic Triton dW reduction."""
import torch
import triton
import triton.language as tl

@triton.jit
def _forward(X,W,Y,INV,H:tl.constexpr,EPS:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0);c=tl.arange(0,D);mask=c<H
    x=tl.load(X+row*H+c,mask,0).to(tl.float32)
    w=tl.load(W+c,mask,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/H+EPS)
    normalized=(x*inv).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+row*H+c,normalized*w,mask);tl.store(INV+row,inv)

@triton.jit
def _backward(X,W,G,INV,DX,P,M:tl.constexpr,H:tl.constexpr,R:tl.constexpr,D:tl.constexpr):
    part=tl.program_id(0);r=part*R+tl.arange(0,R);c=tl.arange(0,D)
    mask=(r[:,None]<M)&(c[None,:]<H)
    x=tl.load(X+r[:,None]*H+c[None,:],mask,0).to(tl.float32)
    g=tl.load(G+r[:,None]*H+c[None,:],mask,0).to(tl.float32)
    w=tl.load(W+c,c<H,0).to(tl.float32)
    inv=tl.load(INV+r,r<M,0)
    weighted=(g*w[None,:]).to(X.dtype.element_ty).to(tl.float32)
    grad_inv=tl.sum(weighted*x,1)
    grad_var=(-0.5*grad_inv)*((inv*inv)*inv)
    dx=weighted*inv[:,None]+(grad_var/H)[:,None]*(2.*x)
    normalized=(x*inv[:,None]).to(X.dtype.element_ty).to(tl.float32)
    dw=(g*normalized).to(W.dtype.element_ty).to(tl.float32)
    tl.store(DX+r[:,None]*H+c[None,:],dx,mask)
    tl.store(P+part*H+c,tl.sum(dw,0),c<H)

@triton.jit
def _reduce_rows(P,OUT,ROWS:tl.constexpr,H:tl.constexpr,R:tl.constexpr,C:tl.constexpr):
    group=tl.program_id(0);r=group*R+tl.arange(0,R)
    c=tl.program_id(1)*C+tl.arange(0,C)
    values=tl.load(P+r[:,None]*H+c[None,:],(r[:,None]<ROWS)&(c[None,:]<H),0)
    tl.store(OUT+group*H+c,tl.sum(values,0),c<H)

class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,eps,optimized):
        if not x.is_cuda or x.device!=weight.device or x.dtype!=weight.dtype:
            raise ValueError('Same-device/dtype CUDA input and weight required; no fallback')
        if x.dtype not in (torch.float32,torch.bfloat16,torch.float16):raise TypeError('Floating point required')
        if weight.ndim!=1 or not x.is_contiguous() or not weight.is_contiguous() or x.shape[-1]!=weight.numel() or x.numel()<1:
            raise ValueError('Contiguous [...,H] input and [H] weight required')
        h=weight.numel()
        if h>8192 or eps<=0:raise ValueError('Supported H<=8192 and positive epsilon required')
        y=torch.empty_like(x);inv=torch.empty(x.numel()//h,device=x.device,dtype=torch.float32)
        _forward[(inv.numel(),)](x,weight,y,inv,h,eps,triton.next_power_of_2(h),enable_fp_fusion=False)
        ctx.save_for_backward(x,weight,inv);ctx.optimized=optimized
        return y
    @staticmethod
    def backward(ctx,dy):
        x,w,inv=ctx.saved_tensors;dy=dy.contiguous();h=w.numel();m=inv.numel()
        r=16 if ctx.optimized and h<=256 else 1;parts=triton.cdiv(m,r)
        dx=torch.empty_like(x);p=torch.empty((parts,h),device=x.device,dtype=torch.float32)
        _backward[(parts,)](x,w,dy,inv,dx,p,m,h,r,triton.next_power_of_2(h),num_warps=4,enable_fp_fusion=False)
        groups=triton.cdiv(parts,128);small=torch.empty((groups,h),device=x.device,dtype=torch.float32)
        dw=torch.empty_like(w)
        _reduce_rows[(groups,triton.cdiv(h,32))](p,small,parts,h,128,32,enable_fp_fusion=False)
        _reduce_rows[(1,triton.cdiv(h,32))](small,dw,groups,h,triton.next_power_of_2(groups),32,enable_fp_fusion=False)
        return dx,dw,None,None

def rms_norm(x,weight,eps=1e-6,*,optimized=False):
    return _RMSNorm.apply(x,weight,eps,optimized)
