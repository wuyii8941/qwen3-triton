"""Explicit multi-row launch candidates; inference-only and HF rounding.

No runtime autotuning here: benchmarks select parameters outside timed regions.
The original training kernels/backward are deliberately left unchanged.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_rows(X,W,Y,M:tl.constexpr,N:tl.constexpr,EPS:tl.constexpr,
              ROWS:tl.constexpr,D:tl.constexpr):
    r=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    d=tl.arange(0,D)
    x=tl.load(X+r[:,None]*N+d[None,:],(r[:,None]<M)&(d[None,:]<N),0).to(tl.float32)
    w=tl.load(W+d,d<N,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,1)/N+EPS)
    z=(x*inv[:,None]).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+r[:,None]*N+d[None,:],z*w[None,:],(r[:,None]<M)&(d[None,:]<N))


def rms_norm_rows(x,weight,eps=1e-6,*,rows=4,num_warps=4):
    if not x.is_cuda or x.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise ValueError('CUDA floating input required')
    if x.ndim<1 or weight.shape!=(x.shape[-1],) or weight.device!=x.device or weight.dtype!=x.dtype:
        raise ValueError('Incompatible weight')
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise ValueError('Inference only')
    if rows not in (1,2,4,8,16) or num_warps not in (4,8):
        raise ValueError('Unsupported launch configuration')
    n=x.shape[-1]
    if not 1<=n<=8192:
        raise ValueError('hidden dimension must be 1..8192')
    x,weight=x.contiguous(),weight.contiguous()
    y=torch.empty_like(x)
    m=x.numel()//n
    if m:
        _rms_rows[(triton.cdiv(m,rows),)](x,weight,y,m,n,eps,rows,
            triton.next_power_of_2(n),num_warps=num_warps,enable_fp_fusion=False)
    return y


@triton.jit
def _rope_flat(X,C,S,Y,H:tl.constexpr,T:tl.constexpr,D:tl.constexpr,
               XB:tl.constexpr,XH:tl.constexpr,XT:tl.constexpr,XD:tl.constexpr,
               TOTAL:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    d=i%D
    t=i//D%T
    h=i//(D*T)%H
    b=i//(D*T*H)
    paired=tl.where(d<D//2,d+D//2,d-D//2)
    base=b*XB+h*XH+t*XT
    x=tl.load(X+base+d*XD,i<TOTAL,0).to(tl.float32)
    xp=tl.load(X+base+paired*XD,i<TOTAL,0).to(tl.float32)
    c=tl.load(C+(b*T+t)*D+d,i<TOTAL,0).to(tl.float32)
    s=tl.load(S+(b*T+t)*D+d,i<TOTAL,0).to(tl.float32)
    a=(x*c).to(Y.dtype.element_ty).to(tl.float32)
    z=(xp*tl.where(d<D//2,-1.,1.)*s).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+i,a+z,i<TOTAL)


def rope_rows(x,cos,sin,*,block=512,num_warps=4):
    if x.ndim!=4 or cos.ndim!=3 or sin.shape!=cos.shape:
        raise ValueError('Expected BHSD input and BSD tables')
    b,h,t,d=x.shape
    if min(b,h,t,d)<1 or d%2 or cos.shape!=(b,t,d):
        raise ValueError('Invalid shape or odd head_dim')
    if not x.is_cuda or x.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise ValueError('CUDA floating input required')
    if any(a.device!=x.device or a.dtype!=x.dtype for a in (cos,sin)):
        raise ValueError('Same-device/dtype tables required')
    if torch.is_grad_enabled() and any(a.requires_grad for a in (x,cos,sin)):
        raise ValueError('Inference only')
    if block not in (128,256,512,1024,2048) or num_warps not in (4,8):
        raise ValueError('Unsupported launch configuration')
    cos,sin=cos.contiguous(),sin.contiguous()
    y=torch.empty(x.shape,device=x.device,dtype=x.dtype)
    _rope_flat[(triton.cdiv(x.numel(),block),)](x,cos,sin,y,h,t,d,*x.stride(),x.numel(),
        block,num_warps=num_warps,enable_fp_fusion=False)
    return y
