"""Causal token-mean CE: shift, LSE, count, mean and backward all in Triton."""
import torch
import triton
import triton.language as tl

@triton.jit
def _partials(X,P,S:tl.constexpr,V:tl.constexpr,PARTS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);part=tl.program_id(1)
    physical=(row//(S-1))*S+row%(S-1)
    c=part*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+physical*V+c,c<V,-float('inf')).to(tl.float32)
    maximum=tl.max(x,0);lse=maximum+tl.log(tl.sum(tl.exp(x-maximum),0))
    tl.store(P+row*PARTS+part,lse)

@triton.jit
def _rows(X,P,LABELS,LSE,LOSSES,S:tl.constexpr,V:tl.constexpr,PARTS:tl.constexpr,IGNORE:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0);physical=(row//(S-1))*S+row%(S-1)
    c=tl.arange(0,D);p=tl.load(P+row*PARTS+c,c<PARTS,-float('inf'))
    maximum=tl.max(p,0);lse=maximum+tl.log(tl.sum(tl.exp(p-maximum),0))
    label=tl.load(LABELS+physical+1)
    valid=(label>=0)&(label<V)
    tl.device_assert(valid|(label==IGNORE),'CE label out of range')
    score=tl.load(X+physical*V+tl.where(valid,label,0),valid,0).to(tl.float32)
    tl.store(LSE+row,lse);tl.store(LOSSES+row,tl.where(label==IGNORE,0.,lse-score))

@triton.jit
def _mean(LOSSES,LABELS,COUNT,OUT,N:tl.constexpr,S:tl.constexpr,IGNORE:tl.constexpr,D:tl.constexpr):
    row=tl.arange(0,D);physical=(row//(S-1))*S+row%(S-1)
    label=tl.load(LABELS+physical+1,row<N,IGNORE)
    count=tl.sum(((row<N)&(label!=IGNORE)).to(tl.int32),0)
    loss=tl.sum(tl.load(LOSSES+row,row<N,0),0)
    tl.store(COUNT,count);tl.store(OUT,loss/count)

@triton.jit
def _backward(X,LABELS,LSE,COUNT,UP,DX,S:tl.constexpr,V:tl.constexpr,IGNORE:tl.constexpr,BLOCK:tl.constexpr):
    physical=tl.program_id(0);col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    valid_row=physical%S<S-1
    row=(physical//S)*(S-1)+physical%S
    label=tl.load(LABELS+physical+1,valid_row,IGNORE)
    x=tl.load(X+physical*V+col,(col<V)&valid_row,0).to(tl.float32)
    lse=tl.load(LSE+row,valid_row,0)
    scale=tl.load(UP).to(tl.float32)/tl.load(COUNT).to(tl.float32)
    value=(tl.exp(x-lse)-(col==label).to(tl.float32))*scale
    tl.store(DX+physical*V+col,tl.where((label==IGNORE)|(~valid_row),0.,value),col<V)

class _CausalCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logits,labels,ignore_index,optimized):
        if not logits.is_cuda or logits.device!=labels.device or logits.ndim!=3 or labels.shape!=logits.shape[:2]:
            raise ValueError('CUDA [B,S,V] logits and same-device [B,S] labels required')
        if labels.dtype!=torch.int64 or not logits.is_contiguous() or not labels.is_contiguous():raise ValueError('Contiguous logits/int64 labels required')
        if logits.dtype not in (torch.float32,torch.bfloat16,torch.float16):raise TypeError('Floating point logits required')
        b,s,v=logits.shape
        if b<1 or s<2 or v<1:raise ValueError('Nonempty batch/vocabulary and at least2tokens required')
        n=b*(s-1);block=2048 if optimized else 1024;parts=triton.cdiv(v,block)
        p=torch.empty((n,parts),device=logits.device,dtype=torch.float32)
        lse=torch.empty(n,device=logits.device,dtype=torch.float32);losses=torch.empty_like(lse)
        count=torch.empty((),device=logits.device,dtype=torch.int32);loss=torch.empty((),device=logits.device,dtype=torch.float32)
        _partials[(n,parts)](logits,p,s,v,parts,block,enable_fp_fusion=False)
        _rows[(n,)](logits,p,labels,lse,losses,s,v,parts,ignore_index,triton.next_power_of_2(parts),debug=True,enable_fp_fusion=False)
        _mean[(1,)](losses,labels,count,loss,n,s,ignore_index,triton.next_power_of_2(n),enable_fp_fusion=False)
        ctx.save_for_backward(logits,labels,lse,count);ctx.ignore=ignore_index;ctx.block=block
        return loss
    @staticmethod
    def backward(ctx,dy):
        x,labels,lse,count=ctx.saved_tensors;b,s,v=x.shape;dx=torch.empty_like(x)
        _backward[(b*s,triton.cdiv(v,ctx.block))](x,labels,lse,count,dy.contiguous(),dx,s,v,ctx.ignore,ctx.block,enable_fp_fusion=False)
        return dx,None,None,None

def causal_cross_entropy(logits,labels,*,ignore_index=-100,optimized=False):
    return _CausalCE.apply(logits,labels,ignore_index,optimized)
