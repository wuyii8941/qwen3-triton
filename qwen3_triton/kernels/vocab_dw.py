"""Fuse per-data-rank dW GEMMs, BF16 rounding and the FP32 mean.

The BF16 boundary after each data rank's GEMM is explicit. Accumulate the
four rounded contributions in FP32, in rank order, without writing separate
BF16 partial weight gradients.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _vocab_dw_mean(DZ,H,DW,V:tl.constexpr,K:tl.constexpr,R:tl.constexpr,WORLD:tl.constexpr,
                   BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GROUP:tl.constexpr):
    pid=tl.program_id(0);nm=tl.cdiv(V,BM);nn=tl.cdiv(K,BN)
    group=pid//(GROUP*nn);first=group*GROUP;height=tl.minimum(nm-first,GROUP)
    row=first+(pid%(GROUP*nn))%height;col=(pid%(GROUP*nn))//height
    vr=row*BM+tl.arange(0,BM);hc=col*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    total=tl.zeros((BM,BN),tl.float32)
    for source in range(WORLD):
        acc=tl.zeros((BM,BN),tl.float32)
        for block in range(tl.cdiv(R,BK)):
            rr=block*BK+kk
            dz=tl.load(DZ+(source*R+rr[None,:])*V+vr[:,None],(vr[:,None]<V)&(rr[None,:]<R),0)
            hidden=tl.load(H+(source*R+rr[:,None])*K+hc[None,:],(rr[:,None]<R)&(hc[None,:]<K),0)
            acc=tl.dot(dz,hidden,acc,input_precision='ieee')
        rounded=acc.to(tl.bfloat16).to(tl.float32)
        if source==0:total=rounded*(1./WORLD)
        else:total=total+rounded*(1./WORLD)
    tl.store(DW+vr[:,None]*K+hc[None,:],total,(vr[:,None]<V)&(hc[None,:]<K))


def weight_gradient_mean(dz,hidden,rows_per_rank,world,config=(64,64,32,8,4,3)):
    if (dz.dtype!=torch.bfloat16 or hidden.dtype!=torch.bfloat16 or not dz.is_cuda or
            dz.device!=hidden.device or not dz.is_contiguous() or not hidden.is_contiguous() or
            dz.shape[0]!=hidden.shape[0] or dz.shape[0]!=rows_per_rank*world):
        raise ValueError('Contiguous same-device BF16 [world*R,V] dZ and [world*R,H] hidden required')
    if world!=4 or rows_per_rank!=512:raise ValueError('Vocabulary weight gradients require four ranks and 512 rows')
    bm,bn,bk,group,warps,stages=config
    v,k=dz.shape[1],hidden.shape[1];out=torch.empty((v,k),device=dz.device,dtype=torch.float32)
    _vocab_dw_mean[(triton.cdiv(v,bm)*triton.cdiv(k,bn),)](
        dz,hidden,out,v,k,rows_per_rank,world,bm,bn,bk,group,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return out
