"""Measured SM86 reference-order BF16 linears, entirely in Triton.

Unsupported shapes/layouts return None so the caller retains its existing
Triton route. Partition/workspace accumulation is FP32; outputs remain BF16.
"""
import torch
import triton
import triton.language as tl
from ..runtime.memory import raw_empty
from ..kernels.split_k import _split_gemm


def strategy(m,n,k):
    if m==1:
        if n==4096 and k in (4096,12288):return ('interleaved',2)
        if k==4096 and n in (1024,12288):return ('gemv',32 if n==1024 else 16)
    if m==17:
        if (n,k)==(1024,4096):return ('interleaved',2)
        if (n,k)==(12288,4096):return ('split',3)
    if m==65:
        if (n,k)==(4096,4096):return ('split',3)
        if (n,k)==(1024,4096):return ('split',5)
        if (n,k)==(4096,12288):return ('split',7)
    return None


@triton.jit
def _inference_interleaved_linear(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr):
    row=tl.program_id(1)*16+tl.arange(0,16);col=tl.program_id(0)*32+tl.arange(0,32);kk=tl.arange(0,32)
    a0=tl.zeros((16,32),tl.float32);a1=tl.zeros((16,32),tl.float32)
    for step in range(tl.cdiv(K,64)):
        k0=step*64+kk;k1=k0+32
        x0=tl.load(X+row[:,None]*K+k0[None,:],(row[:,None]<M)&(k0[None,:]<K),0)
        x1=tl.load(X+row[:,None]*K+k1[None,:],(row[:,None]<M)&(k1[None,:]<K),0)
        w0=tl.load(W+col[None,:]*K+k0[:,None],(col[None,:]<N)&(k0[:,None]<K),0)
        w1=tl.load(W+col[None,:]*K+k1[:,None],(col[None,:]<N)&(k1[:,None]<K),0)
        a0=tl.dot(x0,w0,a0,input_precision='ieee');a1=tl.dot(x1,w1,a1,input_precision='ieee')
    tl.store(Y+row[:,None]*N+col[None,:],a0+a1,(row[:,None]<M)&(col[None,:]<N))


@triton.jit
def _inference_reduce_parts(P,Y,SIZE:tl.constexpr,SPLITS:tl.constexpr,B:tl.constexpr):
    offset=tl.program_id(0)*B+tl.arange(0,B);value=tl.full((B,),0.,tl.float32)
    for part in tl.static_range(SPLITS):
        value=value+tl.load(P+part*SIZE+offset,offset<SIZE,0)
    tl.store(Y+offset,value,offset<SIZE)


@triton.jit
def _inference_reference_gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,LANES:tl.constexpr,UNROLL:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN);lane=tl.arange(0,LANES)
    acc=tl.zeros((BN,LANES),tl.float32)
    for block in range(tl.cdiv(K,LANES*UNROLL)):
        for sub in tl.static_range(UNROLL):
            k=(block*UNROLL+sub)*LANES+lane
            x=tl.load(X+k,k<K,0).to(tl.float32)
            w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
            acc=tl.fma(w,x[None,:],acc)
    steps:tl.constexpr=5 if LANES==32 else 4
    for step in tl.static_range(steps):
        other=tl.broadcast_to((lane^(1<<(steps-1-step)))[None,:],(BN,LANES))
        acc=acc+tl.gather(acc,other,1)
    value=tl.gather(acc,tl.full((BN,1),0,tl.int32),1).reshape((BN,))
    tl.store(Y+n,value,n<N)


def linear(x,weight,*,gemv=True):
    """Return a supported reference-order result, or None without GPU work."""
    if x.ndim<2 or weight.ndim!=2 or not x.is_cuda or x.dtype!=torch.bfloat16 or weight.dtype!=x.dtype or weight.device!=x.device or not x.is_contiguous() or not weight.is_contiguous() or x.shape[-1]!=weight.shape[1] or x.shape[-1]==0:return None
    k=x.shape[-1];m=x.numel()//k;n=weight.shape[0];choice=strategy(m,n,k)
    if choice is None or (choice[0]=='gemv' and not gemv):return None
    y=raw_empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    kind,value=choice
    if kind=='interleaved':
        _inference_interleaved_linear[(triton.cdiv(n,32),triton.cdiv(m,16))](x,weight,y,m,n,k,num_warps=4,num_stages=2,enable_fp_fusion=False)
    elif kind=='gemv':
        bn=4 if value==32 else 8
        _inference_reference_gemv[(triton.cdiv(n,bn),)](x,weight,y,n,k,bn,value,8,num_warps=4,enable_fp_fusion=False)
    else:
        partial=raw_empty((value,m,n),device=x.device,dtype=torch.float32)
        tm=128 if m==65 else 32;tn=64
        _split_gemm[(triton.cdiv(m,tm)*triton.cdiv(n,tn),value)](x,weight,partial,m,n,k,k,1,1,k,tm,tn,64,1,value,num_warps=4,num_stages=2)
        _inference_reduce_parts[(triton.cdiv(m*n,256),)](partial,y,m*n,value,256,enable_fp_fusion=False)
    return y


@triton.jit
def _inference_reference_norm(X,W,Y,H:tl.constexpr,WIDTH:tl.constexpr,EPS:tl.constexpr,STEPS:tl.constexpr):
    row=tl.program_id(0);lane=tl.arange(0,WIDTH)
    a=tl.full((WIDTH,),0.,tl.float32);b=a;c=a;d=a
    for start in range(H//(WIDTH*4)):
        off=row*H+start*WIDTH*4+lane*4
        x0=tl.load(X+off).to(tl.float32);x1=tl.load(X+off+1).to(tl.float32)
        x2=tl.load(X+off+2).to(tl.float32);x3=tl.load(X+off+3).to(tl.float32)
        a=a+x0*x0;b=b+x1*x1;c=c+x2*x2;d=d+x3*x3
    value=((a+b)+c)+d
    for step in tl.static_range(STEPS):
        value=value+tl.gather(value,lane^(WIDTH//2>>step),0)
    mean=tl.sum(tl.where(lane==0,value,0.),0)*(1./H)
    inv=tl.rsqrt(mean+EPS);col=tl.arange(0,H)
    x=tl.load(X+row*H+col).to(tl.float32);w=tl.load(W+col).to(tl.float32)
    normal=(x*inv).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+row*H+col,normal*w)


def short_norm(x,weight,eps):
    """Native2.10 short-row block-x tree; None outside the measured domain."""
    if x.ndim<2 or x.shape[-1]!=4096 or not x.is_cuda or x.dtype!=torch.bfloat16 or not x.is_contiguous() or weight.shape!=(4096,) or weight.dtype!=x.dtype or weight.device!=x.device or not weight.is_contiguous():return None
    rows=x.numel()//4096
    if not 1<=rows<16:return None
    # Reduce.cuh: max512threads, block_height=min(last_pow2(rows),16).
    height=1<<(rows.bit_length()-1);width=512//height
    y=raw_empty(x.shape,device=x.device,dtype=x.dtype)
    _inference_reference_norm[(rows,)](x,weight,y,4096,width,eps,width.bit_length()-1,num_warps=4,enable_fp_fusion=False)
    return y
