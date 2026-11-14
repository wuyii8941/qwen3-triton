"""Unqualified Triton attention controls matching native BF16 cast locations.

Algorithm audit: PyTorch2.10 Flash submodule979702c87a8713a8e0a5e9fee122b90d2ef13be5.
Sources and original BSD3-Clause notice retained in flash_source_audit_001/.
The Triton implementation below is new; it neither calls nor builds Flash.
Native reductions need not be bitwise reproducible in a different compiler.
"""
import math,struct
import torch
import triton
import triton.language as tl
from qwen3_triton.kernels.attention_kernels import _attention_delta_kernel,_attention_kv_head_reduce_kernel

@triton.jit
def _native_precision_forward(Q,K,V,O,L,HQ:tl.constexpr,HK:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
                              SCALE:tl.constexpr,SL2:tl.constexpr,M:tl.constexpr,N:tl.constexpr,
                              LANE_SUM:tl.constexpr,RECIPROCAL:tl.constexpr,LSE_FMA:tl.constexpr,XOR_SUM:tl.constexpr,SUM_FMA:tl.constexpr):
    qi=tl.program_id(0)*M+tl.arange(0,M);head=tl.program_id(1)
    kh=head//HQ*HK+(head%HQ)//(HQ//HK);d=tl.arange(0,D)
    q=tl.load(Q+(head*S+qi[:,None])*D+d[None,:],qi[:,None]<S,0)
    maximum=tl.full((M,),-float('inf'),tl.float32);acc=tl.zeros((M,D),tl.float32)
    if LANE_SUM:
        total=tl.zeros((M,4),tl.float32)
    else:
        total=tl.zeros((M,),tl.float32)
    end=tl.minimum(tl.cdiv(S,N),tl.cdiv((tl.program_id(0)+1)*M,N))
    for step in range(end):
        ki=(end-1-step)*N+tl.arange(0,N)
        off=(kh*S+ki[:,None])*D+d[None,:]
        k=tl.load(K+off,ki[:,None]<S,0);v=tl.load(V+off,ki[:,None]<S,0)
        score=tl.dot(q,tl.trans(k))
        score=tl.where((ki[None,:]<S)&(ki[None,:]<=qi[:,None]),score,-float('inf'))
        new=tl.maximum(maximum,tl.max(score,1));alpha=tl.exp2((maximum-new)*SL2)
        p=tl.exp2(score*SL2-new[:,None]*SL2)
        if LANE_SUM:
            if not SUM_FMA:total=total*alpha[:,None]
            lanes=tl.arange(0,4)
            for index in tl.static_range(N//8):
                ix=tl.broadcast_to((lanes*2+index*8)[None,:],(M,4))
                if SUM_FMA and index==0:total=tl.fma(total,alpha[:,None],tl.gather(p,ix,1))
                else:total=total+tl.gather(p,ix,1)
                total=total+tl.gather(p,ix+1,1)
        else:
            total=total*alpha+tl.sum(p,1)
        acc=tl.dot(p.to(q.dtype),v,acc*alpha[:,None]);maximum=new
    if LANE_SUM:
        if XOR_SUM:
            lanes=tl.arange(0,4)
            total=total+tl.gather(total,tl.broadcast_to((lanes^2)[None,:],(M,4)),1)
            total=total+tl.gather(total,tl.broadcast_to((lanes^1)[None,:],(M,4)),1)
            total=tl.sum(tl.where(lanes[None,:]==0,total,0.),1)
        else:total=tl.sum(total,1)
    if RECIPROCAL==1:
        inverse=tl.div_rn(1.,total)
    elif RECIPROCAL==2:
        inverse=tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;",constraints="=f,f",args=[total],dtype=tl.float32,is_pure=True,pack=1)
    else:
        inverse=1./total
    tl.store(O+(head*S+qi[:,None])*D+d[None,:],acc*inverse[:,None],qi[:,None]<S)
    lse=tl.fma(maximum,SCALE,tl.log(total)) if LSE_FMA else maximum*SCALE+tl.log(total)
    tl.store(L+head*S+qi,lse,qi<S)

@triton.jit
def _native_precision_dq(Q,K,V,G,L,DELTA,DQ,HQ:tl.constexpr,HK:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
                         SCALE:tl.constexpr,SL2:tl.constexpr,M:tl.constexpr,N:tl.constexpr):
    qi=tl.program_id(0)*M+tl.arange(0,M);head=tl.program_id(1)
    kh=head//HQ*HK+(head%HQ)//(HQ//HK);d=tl.arange(0,D);off=(head*S+qi[:,None])*D+d[None,:]
    q=tl.load(Q+off,qi[:,None]<S,0);g=tl.load(G+off,qi[:,None]<S,0)
    lse=tl.load(L+head*S+qi,qi<S,0);delta=tl.load(DELTA+head*S+qi,qi<S,0)
    acc=tl.zeros((M,D),tl.float32)
    for start in range(0,tl.minimum(S,(tl.program_id(0)+1)*M),N):
        ki=start+tl.arange(0,N);koff=(kh*S+ki[:,None])*D+d[None,:]
        k=tl.load(K+koff,ki[:,None]<S,0);v=tl.load(V+koff,ki[:,None]<S,0)
        p=tl.exp2(tl.dot(q,tl.trans(k))*SL2-lse[:,None]*1.4426950408889634)
        p=tl.where((ki[None,:]<S)&(ki[None,:]<=qi[:,None]),p,0.)
        dp=tl.dot(g,tl.trans(v));ds=(p*(dp-delta[:,None])).to(q.dtype)
        acc=tl.dot(ds,k,acc)
    tl.store(DQ+off,acc*SCALE,qi[:,None]<S)

@triton.jit
def _native_precision_dkdv(Q,K,V,G,L,DELTA,PK,PV,HQ:tl.constexpr,HK:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
                           SCALE:tl.constexpr,SL2:tl.constexpr,M:tl.constexpr,N:tl.constexpr):
    ki=tl.program_id(0)*N+tl.arange(0,N);head=tl.program_id(1)
    kh=head//HQ*HK+(head%HQ)//(HQ//HK);d=tl.arange(0,D);off=(kh*S+ki[:,None])*D+d[None,:]
    k=tl.load(K+off,ki[:,None]<S,0);v=tl.load(V+off,ki[:,None]<S,0)
    dk=tl.zeros((N,D),tl.float32);dv=tl.zeros((N,D),tl.float32)
    first=tl.program_id(0)*N//M
    for step in range(tl.cdiv(S,M)-first):
        qi=(tl.cdiv(S,M)-1-step)*M+tl.arange(0,M);qoff=(head*S+qi[:,None])*D+d[None,:]
        q=tl.load(Q+qoff,qi[:,None]<S,0);g=tl.load(G+qoff,qi[:,None]<S,0)
        lse=tl.load(L+head*S+qi,qi<S,0);delta=tl.load(DELTA+head*S+qi,qi<S,0)
        p=tl.exp2(tl.dot(q,tl.trans(k))*SL2-lse[:,None]*1.4426950408889634)
        p=tl.where((qi[:,None]<S)&(ki[None,:]<S)&(ki[None,:]<=qi[:,None]),p,0.)
        dp=tl.dot(g,tl.trans(v));ds=(p*(dp-delta[:,None])).to(q.dtype)
        dk=tl.dot(tl.trans(ds),q,dk);dv=tl.dot(tl.trans(p.to(q.dtype)),g,dv)
    outoff=(head*S+ki[:,None])*D+d[None,:]
    tl.store(PK+outoff,dk*SCALE,ki[:,None]<S);tl.store(PV+outoff,dv,ki[:,None]<S)

@triton.jit
def _native_order_delta(O,G,DELTA,D:tl.constexpr):
    # Sourceaudit003: D128BF16 uses8logical lanes with8contiguousvalues in
    # each of two64widepages, sequentiallocal sum thenXOR4/2/1.
    row=tl.program_id(0);lane=tl.arange(0,8)
    total=tl.zeros((8,),tl.float32)
    for page in tl.static_range(D//64):
        for index in tl.static_range(8):
            offset=row*D+page*64+lane*8+index
            o=tl.load(O+offset).to(tl.float32);g=tl.load(G+offset).to(tl.float32)
            total=total+o*g
    total=total+tl.gather(total,lane^4,0)
    total=total+tl.gather(total,lane^2,0)
    total=total+tl.gather(total,lane^1,0)
    tl.store(DELTA+row,tl.sum(tl.where(lane==0,total,0.),0))

@triton.jit
def _native_split_dq(Q,K,V,G,L,DELTA,P,HQ:tl.constexpr,HK:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
                         SCALE:tl.constexpr,SL2:tl.constexpr,M:tl.constexpr,N:tl.constexpr,NSPLITS:tl.constexpr):
    qi=tl.program_id(0)*M+tl.arange(0,M);head=tl.program_id(1)
    kh=head//HQ*HK+(head%HQ)//(HQ//HK);d=tl.arange(0,D);off=(head*S+qi[:,None])*D+d[None,:]
    q=tl.load(Q+off,qi[:,None]<S,0);g=tl.load(G+off,qi[:,None]<S,0)
    lse=tl.load(L+head*S+qi,qi<S,0);delta=tl.load(DELTA+head*S+qi,qi<S,0)
    acc=tl.zeros((M,D),tl.float32)
    split=tl.program_id(2)
    end=tl.cdiv(tl.minimum(S,(tl.program_id(0)+1)*M),N)
    for tile in range(split,end,NSPLITS):
        start=tile*N
        ki=start+tl.arange(0,N);koff=(kh*S+ki[:,None])*D+d[None,:]
        k=tl.load(K+koff,ki[:,None]<S,0);v=tl.load(V+koff,ki[:,None]<S,0)
        p=tl.exp2(tl.dot(q,tl.trans(k))*SL2-lse[:,None]*1.4426950408889634)
        p=tl.where((ki[None,:]<S)&(ki[None,:]<=qi[:,None]),p,0.)
        dp=tl.dot(g,tl.trans(v));ds=(p*(dp-delta[:,None])).to(q.dtype)
        partial=tl.dot(ds,k)
        acc=acc+partial
    count=tl.num_programs(1)*S*D
    tl.store(P+split*count+off,acc,qi[:,None]<S)

@triton.jit
def _dq_split_reduce(P,DQ,N:tl.constexpr,NSPLITS:tl.constexpr,SCALE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    total=tl.zeros((BLOCK,),tl.float32)
    for split in tl.static_range(NSPLITS):
        total=total+tl.load(P+split*N+i,i<N,0)
    tl.store(DQ+i,total*SCALE,i<N)

def f32(value):return struct.unpack('f',struct.pack('f',value))[0]

class _NativePrecision(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,fusion,lane_sum,round_kv,reciprocal,stages,lse_fma,native_delta,xor_sum,native_dq,sum_fma):
        b,h,s,d=q.shape;hk=k.shape[1]
        assert q.is_cuda and d==128 and q.dtype==torch.bfloat16 and h%hk==0
        assert k.shape==v.shape==(b,hk,s,d)
        assert all(t.dtype==q.dtype and t.device==q.device and t.is_contiguous() for t in (q,k,v))
        scale=f32(d**-.5);sl2=f32(scale*math.log2(math.e))
        out=torch.empty_like(q);lse=torch.empty((b,h,s),device=q.device,dtype=torch.float32)
        _native_precision_forward[(triton.cdiv(s,64),b*h)](q,k,v,out,lse,h,hk,s,d,scale,sl2,64,64,lane_sum,reciprocal,lse_fma,xor_sum,sum_fma,num_warps=4,num_stages=stages,enable_fp_fusion=fusion)
        ctx.save_for_backward(q,k,v,out,lse);ctx.options=(scale,sl2,fusion,round_kv,native_delta,native_dq)
        return out
    @staticmethod
    def backward(ctx,g):
        q,k,v,out,lse=ctx.saved_tensors;b,h,s,d=q.shape;hk=k.shape[1];scale,sl2,fusion,round_kv,native_delta,native_dq=ctx.options
        g=g.contiguous();delta=torch.empty((b,h,s),device=q.device,dtype=torch.float32)
        if native_delta:_native_order_delta[(b*h*s,)](out,g,delta,d,num_warps=4,enable_fp_fusion=False)
        else:_attention_delta_kernel[(b*h*s,)](out,g,delta,d,d,enable_fp_fusion=False)
        dq=torch.empty_like(q);dk=torch.empty_like(k);dv=torch.empty_like(v)
        pk=torch.empty(q.shape,device=q.device,dtype=q.dtype if round_kv else torch.float32);pv=torch.empty_like(pk)
        if native_dq:
            # Flash deterministic split count. Each split sums keytiles in
            # ascending stride order, then a separate FP32 reduction applies
            # the attention scale and casts once to the output dtype.
            nsplits=triton.cdiv(torch.cuda.get_device_properties(q.device).multi_processor_count,b*h)
            partial=torch.empty((nsplits,*q.shape),device=q.device,dtype=torch.float32)
            _native_split_dq[(triton.cdiv(s,64),b*h,nsplits)](q,k,v,g,lse,delta,partial,h,hk,s,d,scale,sl2,64,64,nsplits,num_warps=4,num_stages=1,enable_fp_fusion=fusion)
            _dq_split_reduce[(triton.cdiv(q.numel(),256),)](partial,dq,q.numel(),nsplits,scale,256,enable_fp_fusion=False)
        else:
            _native_precision_dq[(triton.cdiv(s,64),b*h)](q,k,v,g,lse,delta,dq,h,hk,s,d,scale,sl2,64,64,num_warps=4,num_stages=1,enable_fp_fusion=fusion)
        _native_precision_dkdv[(triton.cdiv(s,64),b*h)](q,k,v,g,lse,delta,pk,pv,h,hk,s,d,scale,sl2,64,64,num_warps=4,num_stages=1,enable_fp_fusion=fusion)
        _attention_kv_head_reduce_kernel[(triton.cdiv(k.numel(),256),)](pk,pv,dk,dv,k.numel(),h//hk,s,d,hk,256)
        return dq,dk,dv,None,None,None,None,None,None,None,None,None,None

def candidate(q,k,v,*,fusion=False,lane_sum=True,round_kv=True,reciprocal=1,stages=1,lse_fma=False,native_delta=False,xor_sum=True,native_dq=False,sum_fma=False):
    return _NativePrecision.apply(q,k,v,fusion,lane_sum,round_kv,reciprocal,stages,lse_fma,native_delta,xor_sum,native_dq,sum_fma)
