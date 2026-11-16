"""Inference-only Triton cache, attention, small-M linear and token kernels."""
import math
import torch
import triton
import triton.language as tl
from ..runtime.memory import raw_empty
from ..kernels.linear import matmul
from ..kernels.math_precision import _hf_norm_forward,_hf_norm_rope_forward


@triton.jit
def _inference_positions(P,N:tl.constexpr,S:tl.constexpr,START:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    tl.store(P+i,START+i%S,i<N)


def positions(batch,sequence,start,device):
    out=raw_empty((batch,sequence),device=device,dtype=torch.int64)
    _inference_positions[(triton.cdiv(out.numel(),256),)](out,out.numel(),sequence,start,256)
    return out


@triton.jit
def _inference_cache_append(K,V,CK,CV,N:tl.constexpr,S:tl.constexpr,H:tl.constexpr,D:tl.constexpr,C:tl.constexpr,START:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    d=i%D;head=(i//D)%H;token=(i//(D*H))%S;batch=i//(D*H*S)
    target=((batch*H+head)*C+START+token)*D+d
    tl.store(CK+target,tl.load(K+i,valid,0),valid)
    tl.store(CV+target,tl.load(V+i,valid,0),valid)


def append(k,v,cache_k,cache_v,start):
    b,s,h,d=k.shape
    _inference_cache_append[(triton.cdiv(k.numel(),256),)](k,v,cache_k,cache_v,k.numel(),s,h,d,cache_k.shape[2],start,256)


@triton.jit
def _inference_cached_attention(Q,K,V,O,S:tl.constexpr,T:tl.constexpr,C:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,START:tl.constexpr,SL2:tl.constexpr,M:tl.constexpr,N:tl.constexpr):
    row=tl.program_id(0)*M+tl.arange(0,M);bh=tl.program_id(1)
    batch=bh//HQ;h=bh%HQ;kh=h//(HQ//HK);d=tl.arange(0,D)
    q=tl.load(Q+((batch*S+row[:,None])*HQ+h)*D+d[None,:],row[:,None]<S,0)
    maximum=tl.full((M,),-float('inf'),tl.float32);total=tl.zeros((M,4),tl.float32)
    acc=tl.zeros((M,D),tl.float32);lanes=tl.arange(0,4)
    end=tl.minimum(tl.cdiv(T,N),tl.cdiv(START+(tl.program_id(0)+1)*M,N))
    for step in range(end):
        token=(end-1-step)*N+tl.arange(0,N)
        offset=((batch*HK+kh)*C+token[:,None])*D+d[None,:]
        k=tl.load(K+offset,token[:,None]<T,0);v=tl.load(V+offset,token[:,None]<T,0)
        score=tl.dot(q,tl.trans(k))
        score=tl.where((token[None,:]<T)&(token[None,:]<=START+row[:,None]),score,-float('inf'))
        new=tl.maximum(maximum,tl.max(score,1))
        # A reverse-scanned tile may contain no visible keys for an early
        # query in a non-aligned chunk. Avoid(-inf)-(-inf), keep its mass0.
        safe_new=tl.where(new==-float('inf'),0.,new)
        alpha=tl.where(maximum==-float('inf'),0.,tl.exp2((maximum-safe_new)*SL2))
        p=tl.exp2(score*SL2-safe_new[:,None]*SL2)
        for index in tl.static_range(N//8):
            ix=tl.broadcast_to((lanes*2+index*8)[None,:],(M,4))
            if index==0:total=tl.fma(total,alpha[:,None],tl.gather(p,ix,1))
            else:total=total+tl.gather(p,ix,1)
            total=total+tl.gather(p,ix+1,1)
        acc=tl.dot(p.to(q.dtype),v,acc*alpha[:,None]);maximum=new
    total=total+tl.gather(total,tl.broadcast_to((lanes^2)[None,:],(M,4)),1)
    total=total+tl.gather(total,tl.broadcast_to((lanes^1)[None,:],(M,4)),1)
    denom=tl.sum(tl.where(lanes[None,:]==0,total,0.),1)
    value=acc*tl.div_rn(1.,denom)[:,None]
    tl.store(O+((batch*S+row[:,None])*HQ+h)*D+d[None,:],value,row[:,None]<S)


@triton.jit
def _inference_decode_partial(Q,K,V,P,L,T:tl.constexpr,C:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,SPLITS:tl.constexpr,CHUNK:tl.constexpr,SCALE:tl.constexpr):
    bh=tl.program_id(0);part=tl.program_id(1);batch=bh//HQ;head=bh%HQ;kh=head//(HQ//HK)
    token=part*CHUNK+tl.arange(0,CHUNK);d=tl.arange(0,D)
    q=tl.load(Q+bh*D+d).to(tl.float32)
    k=tl.load(K+((batch*HK+kh)*C+token[:,None])*D+d[None,:],token[:,None]<T,0).to(tl.float32)
    score=tl.sum(k*q[None,:],1)*SCALE
    score=tl.where(token<T,score,-float('inf'))
    maximum=tl.max(score,0);p=tl.exp(score-maximum);total=tl.sum(p,0)
    v=tl.load(V+((batch*HK+kh)*C+token[:,None])*D+d[None,:],token[:,None]<T,0).to(tl.float32)
    value=tl.sum(p[:,None]*v,0)/total
    tl.store(P+(bh*SPLITS+part)*D+d,value)
    tl.store(L+bh*SPLITS+part,maximum+tl.log(total))


@triton.jit
def _inference_decode_merge(P,L,O,H:tl.constexpr,D:tl.constexpr,SPLITS:tl.constexpr,R:tl.constexpr):
    bh=tl.program_id(0);r=tl.arange(0,R);d=tl.arange(0,D)
    l=tl.load(L+bh*SPLITS+r,r<SPLITS,-float('inf'));p=tl.exp(l-tl.max(l,0));denom=tl.sum(p,0)
    x=tl.load(P+(bh*SPLITS+r[:,None])*D+d[None,:],r[:,None]<SPLITS,0)
    value=tl.sum(x*p[:,None],0)/denom
    tl.store(O+bh*D+d,value)


def attention(q,cache_k,cache_v,start,*,split_decode=True):
    b,s,h,d=q.shape;hk=cache_k.shape[1];capacity=cache_k.shape[2];total=start+s
    out=raw_empty(q.shape,device=q.device,dtype=q.dtype)
    if s==1 and split_decode:
        chunk=256;splits=triton.cdiv(total,chunk)
        partial=raw_empty((b*h,splits,d),device=q.device,dtype=torch.float32)
        lse=raw_empty((b*h,splits),device=q.device,dtype=torch.float32)
        _inference_decode_partial[(b*h,splits)](q,cache_k,cache_v,partial,lse,total,capacity,h,hk,d,splits,chunk,d**-.5,num_warps=8,enable_fp_fusion=False)
        _inference_decode_merge[(b*h,)](partial,lse,out,b*h,d,splits,triton.next_power_of_2(splits),num_warps=4)
    else:
        # Preserve the trained candidate's BF16 rounding and reverse tile order.
        import struct
        scale=struct.unpack('f',struct.pack('f',d**-.5))[0]
        sl2=struct.unpack('f',struct.pack('f',scale*math.log2(math.e)))[0]
        _inference_cached_attention[(triton.cdiv(s,64),b*h)](q,cache_k,cache_v,out,s,total,capacity,h,hk,d,start,sl2,64,64,num_warps=4,num_stages=2,enable_fp_fusion=False)
    return out


@triton.jit
def _inference_gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,ACCURATE:tl.constexpr):
    row=tl.program_id(1);n=tl.program_id(0)*BN+tl.arange(0,BN);k=tl.arange(0,BK)
    acc=tl.zeros((BN,BK),tl.float64 if ACCURATE else tl.float32)
    for block in range(tl.cdiv(K,BK)):
        kk=block*BK+k
        x=tl.load(X+row*K+kk,kk<K,0).to(tl.float32)
        w=tl.load(W+n[:,None]*K+kk[None,:],(n[:,None]<N)&(kk[None,:]<K),0).to(tl.float32)
        product=w*x[None,:]
        acc+=product.to(tl.float64) if ACCURATE else product
    tl.store(Y+row*N+n,tl.sum(acc,1),n<N)


def linear(x,weight,*,gemv=True,accurate=False):
    shape=x.shape;m=x.numel()//shape[-1];k=shape[-1];n=weight.shape[0]
    if gemv and m<=8:
        y=raw_empty((*shape[:-1],n),device=x.device,dtype=x.dtype)
        _inference_gemv[(triton.cdiv(n,16),m)](x,weight,y,n,k,16,256,accurate,num_warps=4,enable_fp_fusion=False)
        return y
    return matmul(x.view(m,k),weight.t(),optimized=True).view(*shape[:-1],n)


@triton.jit
def _inference_argmax_parts(X,V,I,N:tl.constexpr,PARTS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);part=tl.program_id(1);i=part*B+tl.arange(0,B)
    x=tl.load(X+row*N+i,i<N,-float('inf')).to(tl.float32);best=tl.max(x,0)
    index=tl.min(tl.where((i<N)&(x==best),i,2147483647),0)
    tl.store(V+row*PARTS+part,best);tl.store(I+row*PARTS+part,index)


@triton.jit
def _inference_select(V,I,IDS,NEXT,DONE,BATCH:tl.constexpr,PARTS:tl.constexpr,R:tl.constexpr,STRIDE:tl.constexpr,POS:tl.constexpr,EOS:tl.constexpr,PAD:tl.constexpr):
    row=tl.program_id(0);r=tl.arange(0,R)
    v=tl.load(V+row*PARTS+r,r<PARTS,-float('inf'));ix=tl.load(I+row*PARTS+r,r<PARTS,2147483647)
    best=tl.max(v,0);token=tl.min(tl.where(v==best,ix,2147483647),0)
    done=tl.load(DONE+row);token=tl.where(done,PAD,token)
    end=tl.full((),False,tl.int1)
    for eos in tl.static_range(len(EOS)):end=end|(token==EOS[eos])
    tl.store(IDS+row*STRIDE+POS,token);tl.store(NEXT+row,token);tl.store(DONE+row,done|end)


def select(logits,ids,next_token,done,position,eos,pad):
    b,n=logits.shape;parts=triton.cdiv(n,1024)
    values=raw_empty((b,parts),device=logits.device,dtype=torch.float32)
    indices=raw_empty((b,parts),device=logits.device,dtype=torch.int32)
    _inference_argmax_parts[(b,parts)](logits,values,indices,n,parts,1024)
    _inference_select[(b,)](values,indices,ids,next_token,done,b,parts,triton.next_power_of_2(parts),ids.shape[1],position,tuple(eos),pad)


def rms_norm(x,weight,eps):
    # Training's wrapper requires>=16 rows; inference also needs1/8 rows.
    # This kernel's independent per-row reduction does not require that limit.
    y=raw_empty(x.shape,device=x.device,dtype=x.dtype)
    inv=raw_empty((x.numel()//weight.numel(),),device=x.device,dtype=torch.float32)
    _hf_norm_forward[(inv.numel(),)](x,weight,y,inv,weight.numel(),eps,triton.next_power_of_2(weight.numel()),enable_fp_fusion=False)
    return y


def norm_rope(x,weight,cos,sin,eps):
    h=x.shape[-1];rows=x.numel()//h
    y=raw_empty(x.shape,device=x.device,dtype=x.dtype)
    inv=raw_empty((rows,),device=x.device,dtype=torch.float32)
    _hf_norm_rope_forward[(rows,)](x,weight,cos,sin,y,inv,h,x.shape[2],eps,triton.next_power_of_2(h),enable_fp_fusion=False)
    return y
