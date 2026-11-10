"""Small-M row-major linear, with exact-weight and optional W8A16 variants.

The INT8 variant is approximate relative to the original checkpoint. Its
kernel oracle is an explicit low-precision dequantization, not full-model
quality. Intended first for the large Qwen LM head; never used in training.
"""
import torch
import triton
import triton.language as tl
from .weight_only_gemv import GroupwiseInt8


@triton.jit
def _linear_decode(X,W,S,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,
                   GROUP:tl.constexpr,KG:tl.constexpr,QUANTIZED:tl.constexpr,
                   M:tl.constexpr,SHARE_BATCH:tl.constexpr):
    # Large vocabulary can exceed CUDA's grid.y limit with one row/program.
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    m=tl.program_id(1)
    if QUANTIZED:
        g=tl.arange(0,KG)
        j=tl.arange(0,GROUP)
        k=g[:,None]*GROUP+j[None,:]
        w=tl.load(W+n[:,None,None]*K+k[None,:,:],
            (n[:,None,None]<N)&(k[None,:,:]<K),0).to(tl.float32)
        scale=tl.load(S+n[:,None]*(K//GROUP)+g[None,:],
            (n[:,None]<N)&(g[None,:]<K//GROUP),0)
        w=(w*scale[:,:,None]).to(X.dtype.element_ty).to(tl.float32)
        w=tl.reshape(w,(BN,KG*GROUP))
    else:
        k=tl.arange(0,KG)
        w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
    flat_k=tl.arange(0,KG*GROUP)
    if SHARE_BATCH:
        # Read/dequantize the large weight tile once, reuse it across tokens.
        # Fixed small-M unrolling avoids another DRAM pass for each batch row.
        for row in tl.static_range(0,M):
            x=tl.load(X+row*K+flat_k,flat_k<K,0).to(tl.float32)
            value=tl.sum(w*x[None,:],1)
            tl.store(Y+row*N+n,value,n<N)
    else:
        x=tl.load(X+m*K+flat_k,flat_k<K,0).to(tl.float32)
        value=tl.sum(w*x[None,:],1)
        tl.store(Y+m*N+n,value,n<N)


def linear_decode(x,weight,*,block_n=4,num_warps=4,share_batch=False):
    if torch.is_grad_enabled():
        raise ValueError('Inference only')
    quantized=isinstance(weight,GroupwiseInt8)
    w=weight.values if quantized else weight
    dtype=weight.dtype if quantized else w.dtype
    if x.ndim!=2 or w.ndim!=2 or x.shape[1]!=w.shape[1]:
        raise ValueError('Expected X [M,K], W [N,K]')
    m,k=x.shape
    n=w.shape[0]
    if not 1<=m<=8 or not 1<=k<=8192 or n<1:
        raise ValueError('Supported M=1..8, K=1..8192, N>0')
    if not x.is_cuda or x.dtype not in (torch.float16,torch.bfloat16) or x.dtype!=dtype:
        raise ValueError('CUDA FP16/BF16 activations with matching weight dtype required')
    if x.device!=w.device or not x.is_contiguous() or not w.is_contiguous():
        raise ValueError('Contiguous same-device tensors required')
    if block_n not in (1,2,4,8,16) or num_warps not in (4,8):
        raise ValueError('Unsupported launch configuration')
    group=weight.group_size if quantized else 1
    scales=weight.scales if quantized else w
    if quantized and (group not in (16,32,64,128) or k%group or w.dtype!=torch.int8
                      or scales.dtype!=torch.float32 or scales.device!=x.device
                      or scales.shape!=(n,k//group) or not scales.is_contiguous()):
        raise ValueError('Invalid grouped INT8 snapshot')
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    _linear_decode[(triton.cdiv(n,block_n),1 if share_batch else m)](x,w,scales,out,n,k,block_n,group,
        triton.next_power_of_2(k//group),quantized,m,share_batch,num_warps=num_warps,enable_fp_fusion=False)
    return out
