"""Dense causal GQA forward/backward; all arithmetic in explicit Triton kernels."""
import math
import torch
import triton
import triton.language as tl
from .attention_kernels import (_attention_forward_kernel,_attention_delta_kernel,
    _attention_dq_kernel,_attention_dkdv_fused_kernel,_attention_kv_head_reduce_kernel)

@triton.jit
def _cast(X,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(Y+i,tl.load(X+i,i<N,0),i<N)

def cast_tensor(x,dtype):
    if not x.is_cuda or not x.is_contiguous():raise ValueError('Contiguous CUDA tensor required')
    out=torch.empty(x.shape,device=x.device,dtype=dtype)
    _cast[(triton.cdiv(x.numel(),512),)](x,out,x.numel(),512)
    return out

class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,scale,optimized):
        if q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape or q.shape[0]!=k.shape[0] or q.shape[2:]!=k.shape[2:]:raise ValueError('Expected equal-length [B,H,S,D] Q and grouped K/V')
        if not q.is_cuda or any(t.device!=q.device or t.dtype!=q.dtype or not t.is_contiguous() for t in (q,k,v)):
            raise ValueError('Same-device/dtype contiguous CUDA QKV required; no fallback')
        if q.dtype not in (torch.float32,torch.bfloat16,torch.float16):raise TypeError('Floating point QKV required')
        b,hq,s,d=q.shape;hk=k.shape[1]
        if min(b,hq,hk,s,d)<1 or hq%hk or d>256:raise ValueError('Nonempty supported GQA with D<=256 required')
        bm,bn=(32,32) if optimized else (16,32);bd=max(32,triton.next_power_of_2(d))
        accumulator=torch.empty(q.shape,device=q.device,dtype=torch.float32);lse=torch.empty((b,hq,s),device=q.device,dtype=torch.float32)
        # Both arms use the same IEEE FP32 dot policy. Optimization changes tiling.
        _attention_forward_kernel[(b*hq*triton.cdiv(s,bm),)](q,k,v,accumulator,lse,hq,hk,s,d,scale,
            BLOCK_M=bm,BLOCK_N=bn,BLOCK_D=bd,MAX_SEQ_LEN=s,DOT_PRECISION='ieee',num_warps=4,num_stages=1)
        ctx.save_for_backward(q,k,v,lse,accumulator);ctx.scale=scale;ctx.bm=bm;ctx.bn=bn
        return cast_tensor(accumulator,q.dtype)
    @staticmethod
    def backward(ctx,dy):
        q,k,v,lse,out=ctx.saved_tensors;dy=dy.contiguous();b,hq,s,d=q.shape;hk=k.shape[1];bd=max(32,triton.next_power_of_2(d));bm=ctx.bm;bn=ctx.bn
        delta=torch.empty((b,hq,s),device=q.device,dtype=torch.float32)
        _attention_delta_kernel[(b*hq*s,)](out,dy,delta,d,bd,enable_fp_fusion=False)
        dq=torch.empty(q.shape,device=q.device,dtype=torch.float32)
        _attention_dq_kernel[(b*hq*triton.cdiv(s,bm),)](q,k,v,dy,lse,delta,dq,hq,hk,s,d,ctx.scale,
            BLOCK_M=bm,BLOCK_N=bn,BLOCK_D=bd,MAX_SEQ_LEN=s,DOT_PRECISION='ieee',num_warps=4,num_stages=1)
        kp=torch.empty_like(dq);vp=torch.empty_like(dq);dk=torch.empty_like(k);dv=torch.empty_like(v)
        _attention_dkdv_fused_kernel[(b*hq*triton.cdiv(s,bn),)](q,k,v,dy,lse,delta,kp,vp,hq,hk,s,d,ctx.scale,bm,bn,bd,'ieee',num_warps=4,num_stages=1)
        _attention_kv_head_reduce_kernel[(triton.cdiv(k.numel(),256),)](kp,vp,dk,dv,k.numel(),hq//hk,s,d,hk,256)
        return cast_tensor(dq,q.dtype),dk,dv,None,None

def causal_attention(q,k,v,*,scale=None,optimized=False):
    return _Attention.apply(q,k,v,scale if scale is not None else 1./math.sqrt(q.shape[-1]),optimized)
