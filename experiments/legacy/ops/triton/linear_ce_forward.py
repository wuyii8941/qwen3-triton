"""Linear + partial logsumexp fused in GEMM tiles, never stores [N,V].

Each vocabulary tile emits one FP32 LSE per token and its target logit.
Only [N,ceil(V/128)] summaries reach global memory. Native summary reduction
finishes CE; backward can recompute chunks without gradient filtering.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _linear_lse(X,W,T,P,Y,N:tl.constexpr,H:tl.constexpr,V:tl.constexpr,
                NT:tl.constexpr,BM:tl.constexpr=16,BN:tl.constexpr=128,BK:tl.constexpr=32):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    h=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for block in range(tl.cdiv(H,BK)):
        hi=block*BK+h
        x=tl.load(X+rows[:,None]*H+hi[None,:],(rows[:,None]<N)&(hi[None,:]<H),0)
        w=tl.load(W+cols[None,:]*H+hi[:,None],(cols[None,:]<V)&(hi[:,None]<H),0)
        acc+=tl.dot(x,w,input_precision='ieee')
    # Match the low-precision Linear output before the FP32 CE upcast.
    acc=acc.to(X.dtype.element_ty).to(tl.float32)
    z=tl.where(cols[None,:]<V,acc,-float('inf'))
    mx=tl.max(z,1)
    lse=mx+tl.log(tl.sum(tl.exp(z-mx[:,None]),1))
    target=tl.load(T+rows,rows<N,-100)
    target_logit=tl.sum(tl.where(cols[None,:]==target[:,None],acc,0.),1)
    tl.store(P+rows*NT+tl.program_id(1),lse,rows<N)
    tl.store(Y+rows*NT+tl.program_id(1),target_logit,rows<N)


def linear_ce_forward(hidden,weight,labels,ignore_index=-100):
    if hidden.ndim!=2 or weight.ndim!=2 or not hidden.is_cuda:
        raise ValueError('CUDA hidden [N,H], weight [V,H] required')
    if hidden.dtype not in (torch.float16,torch.bfloat16,torch.float32) or hidden.dtype!=weight.dtype:
        raise ValueError('Shared FP16/BF16/FP32 dtype required')
    n,h=hidden.shape;v,wh=weight.shape
    if h!=wh or labels.shape!=(n,):raise ValueError('Invalid dimensions')
    nt=triton.cdiv(v,128)
    partial=torch.empty((n,nt),device=hidden.device,dtype=torch.float32)
    target=torch.empty_like(partial)
    _linear_lse[(triton.cdiv(n,16),nt)](hidden.contiguous(),weight.contiguous(),labels.contiguous(),
        partial,target,n,h,v,nt,num_warps=4)
    loss=partial.logsumexp(-1)-target.sum(-1)
    valid=labels!=ignore_index
    return torch.where(valid,loss,0.).sum()/valid.sum()
