"""Experimental FASA-C building blocks for head_dim=128, one-token decode.

No full GQA KV expansion, no gathered [Hq,budget,D] K/V intermediates.
The compact proxy K IS per query head: report its additional storage honestly.
Native torch.topk remains part of the measured selector cost.
All public entry points are inference-only. Valid-prefix lengths live on GPU,
so the same CUDA Graph can replay at different decode positions.
"""
import torch
import triton
import triton.language as tl
from ops.triton.decode_split_kv import _merge


@triton.jit
def _pack(K, PAIRS, POS, PROXY, HQ: tl.constexpr, HK: tl.constexpr,
          CAP: tl.constexpr, COUNT: tl.constexpr, F: tl.constexpr,
          KB: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr,
          KD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = i % (2 * F)
    h = i // (2 * F) % HQ
    t = i // (2 * F * HQ) % COUNT
    b = i // (2 * F * HQ * COUNT)
    # Grid's second dimension is not needed; batch bound passed in grid size.
    pos = tl.load(POS + t)
    d = tl.load(PAIRS + h * F + c % F) + (c // F) * 64
    x = tl.load(K + b*KB + pos*KT + (h//(HQ//HK))*KH + d*KD)
    tl.store(PROXY + ((b*HQ + h)*CAP + pos)*(2*F) + c, x)


@triton.jit
def _scores(Q, P, PAIRS, USED, SCORES, HQ: tl.constexpr, CAP: tl.constexpr,
            F: tl.constexpr, QB: tl.constexpr, QH: tl.constexpr,
            QD: tl.constexpr, BN: tl.constexpr):
    head = tl.program_id(0)
    b, h = head // HQ, head % HQ
    n = tl.program_id(1)*BN + tl.arange(0, BN)
    c = tl.arange(0, 2*F)
    d = tl.load(PAIRS + h*F + c % F) + (c//F)*64
    q = tl.load(Q + b*QB + h*QH + d*QD).to(tl.float32)
    used = tl.load(USED+b)
    valid = (n < CAP) & (n < used)
    k = tl.load(P + (head*CAP+n[:,None])*(2*F)+c[None,:], valid[:,None], 0).to(tl.float32)
    scores = tl.sum(k*q[None,:], 1)
    tl.store(SCORES + head*CAP+n, tl.where(valid, scores, -float('inf')), n<CAP)


@triton.jit
def _indexed_partial(Q, K, V, IDX, USED, P, L, O,
        HQ: tl.constexpr, HK: tl.constexpr, CAP: tl.constexpr, BUDGET: tl.constexpr,
        QB: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
        KB: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
        VB: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
        SCALE: tl.constexpr, SPLITS: tl.constexpr, BN: tl.constexpr):
    head, split = tl.program_id(0), tl.program_id(1)
    b, h = head//HQ, head%HQ
    kh = h//(HQ//HK)
    d = tl.arange(0,128)
    q = tl.load(Q+b*QB+h*QH+d*QD).to(tl.float32)
    used = tl.load(USED+b)
    mx = -float('inf')
    den = 0.
    acc = tl.zeros((128,), tl.float32)
    tiles = tl.cdiv(BUDGET, BN)
    for tile in range(tiles*split//SPLITS, tiles*(split+1)//SPLITS):
        n = tile*BN+tl.arange(0,BN)
        pos = tl.load(IDX+head*BUDGET+n, n<BUDGET, 0)
        valid = (n<BUDGET) & (pos>=0) & (pos<used) & (pos<CAP)
        k = tl.load(K+b*KB+pos[:,None]*KT+kh*KH+d[None,:]*KD,valid[:,None],0).to(tl.float32)
        v = tl.load(V+b*VB+pos[:,None]*VT+kh*VH+d[None,:]*VD,valid[:,None],0).to(tl.float32)
        scores = tl.where(valid, tl.sum(k*q[None,:],1)*SCALE, -float('inf'))
        new_mx = tl.maximum(mx,tl.max(scores,0))
        # Empty split/tile must contribute zero, not NaN, at short valid prefixes.
        safe_mx = tl.where(new_mx == -float('inf'), 0., new_mx)
        alpha = tl.exp(mx-safe_mx)
        prob = tl.exp(scores-safe_mx)
        acc = acc*alpha+tl.sum(prob[:,None]*v,0)
        den = den*alpha+tl.sum(prob,0)
        mx = new_mx
    normalized = acc/tl.where(den>0,den,1.)
    if SPLITS==1:
        tl.store(O+head*128+d,normalized)
    else:
        row=head*SPLITS+split
        tl.store(P+row*128+d,normalized)
        tl.store(L+row,tl.where(den>0,mx+tl.log(den),-float('inf')))


def _inference(*tensors):
    if any(not t.is_cuda for t in tensors):
        raise ValueError('CUDA tensors required')
    if any(t.device != tensors[0].device for t in tensors):
        raise ValueError('All tensors must share a device')
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise ValueError('Inference-only experimental kernel')


def validate_pairs(pairs, heads):
    """Setup-only validation (synchronizes); never call inside graph capture."""
    if pairs.ndim != 2 or pairs.shape[0] != heads or pairs.shape[1] not in (8,16,32,64):
        raise ValueError('pairs must be [Hq,F], F=8/16/32/64')
    if pairs.dtype not in (torch.int32, torch.int64) or not pairs.is_contiguous():
        raise ValueError('Contiguous integer pairs required')
    if bool(((pairs<0)|(pairs>=64)).any()) or bool((pairs.sort(-1).values.diff(dim=-1)==0).any()):
        raise ValueError('Distinct split-half pair indices in [0,64) required')


def pack_proxy(k, pairs, positions, proxy):
    """K [B,capacity,Hkv,128] -> persistent [B,Hq,capacity,2F].

    Pairs must pass validate_pairs at setup; positions are checked by the cache.
    Handles arbitrary prefill count; padding lanes are masked in a 1-D grid.
    """
    _inference(k,pairs,positions,proxy)
    b,cap,hk,d=k.shape
    h,f=pairs.shape
    if d!=128 or h%hk or f not in (8,16,32,64) or proxy.shape!=(b,h,cap,2*f):
        raise ValueError('Invalid proxy/GQA shape')
    if not pairs.is_contiguous() or not proxy.is_contiguous() or proxy.dtype!=k.dtype:
        raise ValueError('Contiguous proxy/pairs and shared K/proxy dtype required')
    if (pairs.dtype not in (torch.int32,torch.int64) or positions.dtype not in (torch.int32,torch.int64)
            or positions.ndim!=1 or not positions.is_contiguous() or positions.numel()<1):
        raise ValueError('Integer pairs and nonempty contiguous positions required')
    # 2F*H is a multiple of 16; choose a divisor, so no out-of-range lanes.
    block=min(128,2*f*h)
    while (2*f*h)%block:
        block//=2
    total=b*positions.numel()*h*2*f
    _pack[(total//block,)](k,pairs,positions,proxy,h,hk,cap,positions.numel(),f,
                           *k.stride(),block,num_warps=4)


def proxy_scores(q, proxy, pairs, used):
    _inference(q,proxy,pairs,used)
    if q.ndim!=3 or q.shape[-1]!=128 or not proxy.is_contiguous():
        raise ValueError('Q [B,Hq,128] and contiguous proxy required')
    b,h,_=q.shape
    cap,f=proxy.shape[2],pairs.shape[-1]
    if f not in (8,16,32,64) or proxy.shape!=(b,h,cap,2*f) or pairs.shape!=(h,f) or used.shape!=(b,):
        raise ValueError('Proxy/pairs/used shape mismatch')
    if (proxy.dtype!=q.dtype or not pairs.is_contiguous() or pairs.dtype not in (torch.int32,torch.int64)
            or used.dtype!=torch.int32 or not used.is_contiguous()):
        raise ValueError('Proxy/Q dtype, integer pairs and contiguous int32 used required')
    scores=torch.empty((b,h,cap),device=q.device,dtype=torch.float32)
    _scores[(b*h,triton.cdiv(cap,64))](q,proxy,pairs,used,scores,h,cap,f,*q.stride(),64,num_warps=4)
    return scores


def indexed_attention(q,k,v,indices,used,*,scale=None,splits=8,block_n=64):
    _inference(q,k,v,indices,used)
    if q.ndim!=3 or k.ndim!=4 or v.shape!=k.shape or q.shape[-1]!=128:
        raise ValueError('Q [B,Hq,128], K/V [B,capacity,Hkv,128] required')
    b,h,_=q.shape
    bk,cap,hk,d=k.shape
    if b!=bk or d!=128 or h%hk or indices.ndim!=3 or indices.shape[:2]!=(b,h):
        raise ValueError('Invalid GQA/index shape')
    if q.dtype not in (torch.float16,torch.bfloat16,torch.float32) or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise ValueError('Q/K/V must share FP16/BF16/FP32 dtype')
    if not indices.is_contiguous() or indices.dtype not in (torch.int64,torch.int32):
        raise ValueError('Contiguous integer indices required')
    if used.shape!=(b,) or used.dtype!=torch.int32 or not used.is_contiguous():
        raise ValueError('Contiguous int32 used [B] required')
    budget=indices.shape[-1]
    if not 1<=budget<=cap or not 1<=splits<=64 or block_n not in (16,32,64,128):
        raise ValueError('Invalid budget/splits/block_n')
    splits=min(splits,triton.cdiv(budget,block_n))
    out=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    partial=torch.empty((b,h,splits,128),device=q.device,dtype=torch.float32)
    lse=torch.empty((b,h,splits),device=q.device,dtype=torch.float32)
    _indexed_partial[(b*h,splits)](q,k,v,indices,used,partial,lse,out,h,hk,cap,budget,
        *q.stride(),*k.stride(),*v.stride(),128**-.5 if scale is None else scale,
        splits,block_n,num_warps=4)
    if splits>1:
        _merge[(b*h,)](partial,lse,out,splits,triton.next_power_of_2(splits),num_warps=4)
    return out
