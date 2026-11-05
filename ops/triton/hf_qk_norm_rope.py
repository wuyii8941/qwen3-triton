"""HF-compatible Q/K RMSNorm + RoPE in ONE launch, inference only.

Projection input layout BSHD, contiguous BHSD output for attention/cache.
Every eager cast boundary is preserved; no normalized intermediate allocation.
Unlike the older training kernel, this matches HF's low-precision semantics.
"""
import torch
import triton
import triton.language as tl
from ops import compile_ops


@triton.jit
def _qk_norm_rope(Q,K,WQ,WK,C,S,OQ,OK,T:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
                  D:tl.constexpr,CB:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr,
                  QB:tl.constexpr,QT:tl.constexpr,KB:tl.constexpr,KT:tl.constexpr,
                  POS,CAP:tl.constexpr,DIRECT_K:tl.constexpr,
                  V,OV,VB:tl.constexpr,VH:tl.constexpr,VT:tl.constexpr,DIRECT_V:tl.constexpr):
    row=tl.program_id(0)
    token=row//(HQ+HK)
    head=row%(HQ+HK)
    b=token//T
    t=token%T
    # Uniform branch within a program; Q and K share one launch, not weights.
    if head<HQ:
        X=Q
        W=WQ
        Y=OQ
        h=head
        nh=HQ
        base=b*QB+t*QT+h*D
    else:
        X=K
        W=WK
        Y=OK
        h=head-HQ
        nh=HK
        base=b*KB+t*KT+h*D
    d=tl.arange(0,BLOCK)
    p=tl.where(d<D//2,d+D//2,d-D//2)
    # Packed QKV slices have gaps between tokens; consume strides directly.
    x=tl.load(X+base+d,d<D,0).to(tl.float32)
    xp=tl.load(X+base+p,d<D,0).to(tl.float32)
    w=tl.load(W+d,d<D,0).to(tl.float32)
    wp=tl.load(W+p,d<D,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS)
    # RMSNorm: normalize -> cast -> learned scale -> cast.
    z=(x*inv).to(Y.dtype.element_ty).to(tl.float32)
    zp=(xp*inv).to(Y.dtype.element_ty).to(tl.float32)
    z=(z*w).to(Y.dtype.element_ty).to(tl.float32)
    zp=(zp*wp).to(Y.dtype.element_ty).to(tl.float32)
    tb=tl.where(CB==1,0,b)
    co=tl.load(C+(tb*T+t)*D+d,d<D,0).to(tl.float32)
    si=tl.load(S+(tb*T+t)*D+d,d<D,0).to(tl.float32)
    left=(z*co).to(Y.dtype.element_ty).to(tl.float32)
    right=(zp*tl.where(d<D//2,-1.,1.)*si).to(Y.dtype.element_ty).to(tl.float32)
    out_base=(((b*nh+h)*T+t)*D).to(tl.int64)
    if DIRECT_K:
        if head>=HQ:
            pos=tl.load(POS+t)
            out_base=((b*CAP+pos)*HK+h)*D
            if DIRECT_V:
                # Reuse the K program for its corresponding V cache write.
                # V is not normalized/rotated; preserve its exact input bits.
                value=tl.load(V+b*VB+h*VH+t*VT+d,d<D,0)
                tl.store(OV+out_base+d,value,d<D)
    tl.store(Y+out_base+d,left+right,d<D)


def _launch(q,k,wq,wk,cos,sin,eps,*,k_cache=None,positions=None,v=None,v_cache=None):
    """Internal hot path: caller has already established the input contract."""
    if torch.compiler.is_compiling():
        if k_cache is None:
            return compile_ops.qk_rope(q, k, wq, wk, cos, sin, eps)
        if v_cache is None:
            oq = compile_ops.qk_rope_store_k(q, k, wq, wk, cos, sin, eps, k_cache, positions)
        else:
            oq = compile_ops.qk_rope_store_kv(q, k, wq, wk, cos, sin, eps,
                                            k_cache, positions, v, v_cache)
        # Rebuild the alias outside the custom op; its outputs must be fresh.
        return oq, k_cache[:, :q.shape[1]].transpose(1, 2)
    b,t,hq,d=q.shape
    hk=k.shape[2]
    oq=torch.empty((b,hq,t,d),device=q.device,dtype=q.dtype)
    direct=k_cache is not None
    ok=k_cache if direct else torch.empty((b,hk,t,d),device=k.device,dtype=k.dtype)
    _qk_norm_rope[(b*t*(hq+hk),)](q,k,wq,wk,cos,sin,oq,ok,t,hq,hk,d,cos.shape[0],eps,
        triton.next_power_of_2(d),q.stride(0),q.stride(1),k.stride(0),k.stride(1),
        positions if direct else cos,k_cache.shape[1] if direct else 0,direct,
        v if v_cache is not None else q,v_cache if v_cache is not None else q,
        *(v.stride()[:3] if v_cache is not None else (0,0,0)),v_cache is not None,
        num_warps=4,enable_fp_fusion=False)
    # Initial prefill can attend to this view without copying. During decode
    # the caller instead uses the full cache with a GPU valid-length vector.
    return oq,ok[:,:t].transpose(1,2) if direct else ok


def hf_qk_norm_rope(q,k,wq,wk,cos,sin,eps=1e-6):
    if q.ndim!=4 or k.ndim!=4 or q.shape[:2]!=k.shape[:2] or q.shape[-1]!=k.shape[-1]:
        raise ValueError('Expected BSHD Q/K with matching B,S,D')
    b,t,hq,d=q.shape
    if min(b,t,hq,k.shape[2],d)<=0 or d%2 or d>256:
        raise ValueError('Positive shapes and even head_dim <= 256 required')
    if wq.shape!=(d,) or wk.shape!=(d,) or cos.shape!=sin.shape or cos.shape not in ((1,t,d),(b,t,d)):
        raise ValueError('Invalid norm weights or rotary tables')
    tensors=(q,k,wq,wk,cos,sin)
    if not q.is_cuda or q.dtype not in (torch.float16,torch.bfloat16):
        raise ValueError('CUDA FP16/BF16 required')
    if any(a.device!=q.device or a.dtype!=q.dtype or not a.is_contiguous() for a in tensors):
        raise ValueError('Contiguous, matching device/dtype required')
    if torch.is_grad_enabled() and any(a.requires_grad for a in tensors):
        raise ValueError('Inference only')
    return _launch(q,k,wq,wk,cos,sin,eps)
