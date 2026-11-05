"""Dense CE gradient from saved/recomputed row LSE; no gradient filtering.

Each program writes a vocabulary tile directly in GEMM input dtype, avoiding
the FP32 softmax/probability tensor. GEMMs and LSE remain native references.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gradient(X,LSE,TARGET,COUNT,UP,G,V:tl.constexpr,IGNORE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    target=tl.load(TARGET+row)
    x=tl.load(X+row*V+col,col<V,0).to(tl.float32)
    lse=tl.load(LSE+row)
    scale=tl.load(UP).to(tl.float32)/tl.load(COUNT)
    value=(tl.exp(x-lse)-(col==target).to(tl.float32))*scale
    value=tl.where(target!=IGNORE,value,0.)
    tl.store(G+row*V+col,value,col<V)


def ce_gradient(logits,lse,labels,count,upstream,*,dtype,ignore_index=-100):
    if not logits.is_cuda or not logits.is_contiguous() or logits.ndim!=2:
        raise ValueError('Contiguous CUDA [N,V] logits required')
    out=torch.empty(logits.shape,device=logits.device,dtype=dtype)
    _gradient[(logits.shape[0],triton.cdiv(logits.shape[1],1024))](
        logits,lse,labels,count,upstream,out,logits.shape[1],ignore_index,1024)
    return out
