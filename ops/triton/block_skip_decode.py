"""BLASST-inspired Ampere experiment, not the paper's CUDA implementation.

Algorithm: https://arxiv.org/html/2512.12087v3 (MLSys 2026; first posted 2025).
Every K tile is still read. Only low-score tiles' V loads/softmax/PV are skipped.
Each split maintains its OWN running max; no global-oracle sparsity claim.
Reverse tile order seeds each split with its recent positions. This differs
from the official batched-load CUDA pipeline and must be benchmarked as such.
"""
import math
import torch
import triton
import triton.language as tl
from ops.triton.decode_split_kv import _merge
from ops.triton.frequency_sparse import _inference


@triton.jit
def _partial(Q,K,V,USED,P,L,O,STATS,HQ:tl.constexpr,HK:tl.constexpr,CAP:tl.constexpr,
             QB:tl.constexpr,QH:tl.constexpr,QD:tl.constexpr,
             KB:tl.constexpr,KT:tl.constexpr,KH:tl.constexpr,KD:tl.constexpr,
             VB:tl.constexpr,VT:tl.constexpr,VH:tl.constexpr,VD:tl.constexpr,
             SCALE:tl.constexpr,LOG_THRESHOLD:tl.constexpr,SPLITS:tl.constexpr,BN:tl.constexpr):
    head,split=tl.program_id(0),tl.program_id(1)
    b,h=head//HQ,head%HQ
    kh=h//(HQ//HK)
    used=tl.minimum(tl.load(USED+b),CAP)
    tiles=tl.cdiv(used,BN)
    first,last=tiles*split//SPLITS,tiles*(split+1)//SPLITS
    d=tl.arange(0,128)
    q=tl.load(Q+b*QB+h*QH+d*QD).to(tl.float32)
    mx=-float('inf')
    den=0.
    acc=tl.zeros((128,),tl.float32)
    skipped=0
    for iteration in range(last-first):
        pos=(last-1-iteration)*BN+tl.arange(0,BN)
        valid=pos<used
        k=tl.load(K+b*KB+pos[:,None]*KT+kh*KH+d[None,:]*KD,valid[:,None],0).to(tl.float32)
        score=tl.where(valid,tl.sum(k*q[None,:],1)*SCALE,-float('inf'))
        block_mx=tl.max(score,0)
        new_mx=tl.maximum(mx,block_mx)
        keep=True
        if LOG_THRESHOLD!=-float('inf'):
            keep=block_mx-new_mx>=LOG_THRESHOLD
        if keep:
            v=tl.load(V+b*VB+pos[:,None]*VT+kh*VH+d[None,:]*VD,valid[:,None],0).to(tl.float32)
            alpha=tl.exp(mx-new_mx)
            prob=tl.exp(score-new_mx)
            acc=acc*alpha+tl.sum(prob[:,None]*v,0)
            den=den*alpha+tl.sum(prob,0)
        else:
            skipped+=1
        mx=new_mx
    normalized=acc/tl.where(den>0,den,1.)
    row=head*SPLITS+split
    tl.store(STATS+row*2,skipped)
    tl.store(STATS+row*2+1,last-first)
    if SPLITS==1:
        tl.store(O+head*128+d,normalized)
    else:
        tl.store(P+row*128+d,normalized)
        tl.store(L+row,tl.where(den>0,mx+tl.log(den),-float('inf')))


def block_skip_attention(q,k,v,used,*,threshold=1e-4,splits=16,block_n=64,scale=None):
    """Return (output [B,Hq,128], [skipped,total] counts per head/split).

    threshold=0 is the EXACT dense algorithm (up to normal FP roundoff).
    threshold>0 is APPROXIMATE and never automatically enabled.
    used [B] must be cache-validated with 1<=used<=capacity before capture.
    """
    _inference(q,k,v,used)
    if q.ndim!=3 or k.ndim!=4 or k.shape!=v.shape or q.shape[-1]!=128:
        raise ValueError('Q [B,Hq,128], K/V [B,capacity,Hkv,128] required')
    b,h,_=q.shape
    bk,cap,hk,d=k.shape
    if b!=bk or d!=128 or h%hk or used.shape!=(b,) or used.dtype!=torch.int32 or not used.is_contiguous():
        raise ValueError('Invalid GQA/cache lengths')
    if q.dtype not in (torch.float16,torch.bfloat16,torch.float32) or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise ValueError('Q/K/V must share FP16/BF16/FP32 dtype')
    if not 0<=threshold<=1 or not 1<=splits<=64 or block_n not in (16,32,64,128):
        raise ValueError('Invalid threshold/splits/block_n')
    out=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    partial=torch.empty((b,h,splits,128),device=q.device,dtype=torch.float32)
    lse=torch.empty((b,h,splits),device=q.device,dtype=torch.float32)
    stats=torch.empty((b,h,splits,2),device=q.device,dtype=torch.int32)
    _partial[(b*h,splits)](q,k,v,used,partial,lse,out,stats,h,hk,cap,*q.stride(),
        *k.stride(),*v.stride(),128**-.5 if scale is None else scale,
        math.log(threshold) if threshold else -float('inf'),splits,block_n,num_warps=4)
    if splits>1:_merge[(b*h,)](partial,lse,out,splits,triton.next_power_of_2(splits),num_warps=4)
    return out,stats
