"""Split-KV decode with optional sharing of K/V across a GQA head group.

Inference only, one query token, all cached positions visible. K/V can be
strided views into a static cache: no repeat_kv and no contiguous cache copy.
The merge is a numerically stable softmax merge, NOT an average of partials.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(Q,K,V,P,L,O,HQ:tl.constexpr,HK:tl.constexpr,T:tl.constexpr,
             QB:tl.constexpr,QH:tl.constexpr,QD:tl.constexpr,
             KB:tl.constexpr,KH:tl.constexpr,KT:tl.constexpr,KD:tl.constexpr,
             VB:tl.constexpr,VH:tl.constexpr,VT:tl.constexpr,VD:tl.constexpr,
             SCALE:tl.constexpr,SPLITS:tl.constexpr,G:tl.constexpr,BN:tl.constexpr):
    # G=1: one query head. G=HQ/HK: shared K/V load for a whole GQA group.
    pid=tl.program_id(0)
    split=tl.program_id(1)
    batch=pid//(HQ//G)
    first_head=pid%(HQ//G)*G
    heads=first_head+tl.arange(0,G)
    kv_head=first_head//(HQ//HK)
    d=tl.arange(0,128)
    q=tl.load(Q+batch*QB+heads[:,None]*QH+d[None,:]*QD).to(tl.float32)
    mx=tl.full((G,),-float('inf'),tl.float32)
    den=tl.zeros((G,),tl.float32)
    acc=tl.zeros((G,128),tl.float32)
    tiles=tl.cdiv(T,BN)
    # Partition whole tiles evenly; SPLITS <= tiles ensures nonempty ranges.
    first=tiles*split//SPLITS
    last=tiles*(split+1)//SPLITS
    for tile in range(first,last):
        pos=tile*BN+tl.arange(0,BN)
        k=tl.load(K+batch*KB+kv_head*KH+pos[:,None]*KT+d[None,:]*KD,
                  pos[:,None]<T,0).to(tl.float32)
        v=tl.load(V+batch*VB+kv_head*VH+pos[:,None]*VT+d[None,:]*VD,
                  pos[:,None]<T,0).to(tl.float32)
        scores=tl.sum(q[:,None,:]*k[None,:,:],2)*SCALE
        scores=tl.where(pos[None,:]<T,scores,-float('inf'))
        new_mx=tl.maximum(mx,tl.max(scores,1))
        alpha=tl.exp(mx-new_mx)
        prob=tl.exp(scores-new_mx[:,None])
        acc=acc*alpha[:,None]+tl.sum(prob[:,:,None]*v[None,:,:],1)
        den=den*alpha+tl.sum(prob,1)
        mx=new_mx
    normalized=acc/den[:,None]
    if SPLITS==1:
        tl.store(O+(batch*HQ+heads[:,None])*128+d[None,:],normalized)
    else:
        row=(batch*HQ+heads)*SPLITS+split
        tl.store(P+row[:,None]*128+d[None,:],normalized)
        tl.store(L+row,mx+tl.log(den))


@triton.jit
def _merge(P,L,O,SPLITS:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0)
    s=tl.arange(0,BS)
    d=tl.arange(0,128)
    lse=tl.load(L+row*SPLITS+s,s<SPLITS,-float('inf'))
    w=tl.exp(lse-tl.max(lse,0))
    part=tl.load(P+(row*SPLITS+s[:,None])*128+d[None,:],s[:,None]<SPLITS,0)
    out=tl.sum(w[:,None]*part,0)/tl.sum(w,0)
    tl.store(O+row*128+d,out)


def attention_decode_split(q,k,v,*,splits=8,group_heads=True,block_n=64,scale=None):
    if any(t.ndim!=4 for t in (q,k,v)):
        raise ValueError('Expected rank-4 Q/K/V')
    b,h,t,d=q.shape
    if b<1 or h<1 or t!=1 or d!=128 or k.shape!=v.shape or k.shape[0]!=b or k.shape[-1]!=128:
        raise ValueError('Expected Q [B,Hq,1,128], K/V [B,Hkv,S,128]')
    hk,seq=k.shape[1:3]
    if hk<1 or h%hk or seq<1:
        raise ValueError('Invalid GQA head counts or empty cache')
    if q.dtype not in (torch.float16,torch.bfloat16,torch.float32) or not q.is_cuda:
        raise ValueError('CUDA FP16/BF16/FP32 required')
    if any(x.device!=q.device or x.dtype!=q.dtype for x in (k,v)):
        raise ValueError('Q/K/V must share device/dtype')
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,v)):
        raise ValueError('Inference-only kernel')
    if not isinstance(splits,int) or not 1<=splits<=64 or block_n not in (16,32,64,128):
        raise ValueError('splits must be 1..64; block_n must be 16/32/64/128')
    g=h//hk if group_heads else 1
    if g not in (1,2,4,8):
        raise ValueError('Grouped mode supports GQA ratios 1/2/4/8; use group_heads=False')
    splits=min(splits,triton.cdiv(seq,block_n))
    out=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    if splits>1:
        partial=torch.empty((b,h,splits,128),device=q.device,dtype=torch.float32)
        lse=torch.empty((b,h,splits),device=q.device,dtype=torch.float32)
    else:
        partial,lse=out,out  # P/L are compile-time unused in this specialization.
    _partial[(b*h//g,splits)](q,k,v,partial,lse,out,h,hk,seq,
        q.stride(0),q.stride(1),q.stride(3),
        k.stride(0),k.stride(1),k.stride(2),k.stride(3),
        v.stride(0),v.stride(1),v.stride(2),v.stride(3),
        1/math.sqrt(128) if scale is None else scale,splits,g,block_n,num_warps=4)
    if splits>1:
        _merge[(b*h,)](partial,lse,out,splits,triton.next_power_of_2(splits),num_warps=4)
    return out
