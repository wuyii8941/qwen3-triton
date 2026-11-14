"""Exact BF16 transport packing and FP32 reduction for FSDP contributions.

Packing is allowed only for finite, exact BF16 upcasts. The collective caller
must check the device flag on every rank before using the packed buffer.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_bf16_checked_kernel(X, Y, ERROR, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+offsets,offsets<N,0)
    packed=x.to(tl.bfloat16)
    tl.store(Y+offsets,packed,offsets<N)
    bad=(offsets<N)&((packed.to(tl.float32)!=x)|(tl.abs(x)==float('inf')))
    if tl.sum(bad.to(tl.int32),0)>0:
        tl.atomic_or(ERROR,1)


@triton.jit
def _reduce_bf16_fp32_kernel(X,Y,COLS:tl.constexpr,WORLD:tl.constexpr,
                              AVERAGE:tl.constexpr,BLOCK:tl.constexpr):
    cols=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    ranks=tl.arange(0,triton.next_power_of_2(WORLD))
    values=tl.load(X+ranks[:,None]*COLS+cols[None,:],
                   (ranks[:,None]<WORLD)&(cols[None,:]<COLS),0).to(tl.float32)
    if AVERAGE:values=values*(1./WORLD)
    tl.store(Y+cols,tl.sum(values,0),cols<COLS)


def pack_bf16_checked(value):
    if value.dtype!=torch.float32 or not value.is_cuda or not value.is_contiguous():
        raise ValueError('Contiguous FP32 CUDA input required')
    packed=torch.empty_like(value,dtype=torch.bfloat16)
    error=torch.zeros((),device=value.device,dtype=torch.int32)
    _pack_bf16_checked_kernel[(triton.cdiv(value.numel(),1024),)](
        value,packed,error,value.numel(),1024)
    return packed,error


def reduce_bf16_fp32(received,output,world,average):
    if (received.dtype!=torch.bfloat16 or output.dtype!=torch.float32
            or not received.is_cuda or received.device!=output.device
            or not received.is_contiguous() or not output.is_contiguous()
            or received.numel()!=world*output.numel() or world<2):
        raise ValueError('Matching CUDA BF16 contributions and FP32 output required')
    _reduce_bf16_fp32_kernel[(triton.cdiv(output.numel(),1024),)](
        received,output,output.numel(),world,average,1024,enable_fp_fusion=False)


@triton.jit
def first_checked(X,Y,ERROR,N:tl.constexpr,COLS:tl.constexpr,CHUNK:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(X+offsets,offsets<N,0)
    packed=value.to(tl.bfloat16)
    rank=offsets//COLS;col=offsets%COLS
    tl.store(Y+rank*CHUNK+col,packed,(offsets<N)&(col<CHUNK))
    bad=(offsets<N)&((packed.to(tl.float32)!=value)|(tl.abs(value)==float('inf')))
    if tl.sum(bad.to(tl.int32),0)>0:tl.atomic_or(ERROR,1)


@triton.jit
def later_pack(X,Y,COLS:tl.constexpr,START:tl.constexpr,CHUNK:tl.constexpr,WORLD:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    rank=offsets//CHUNK;col=offsets%CHUNK
    x=tl.load(X+rank*COLS+START+col,offsets<WORLD*CHUNK,0)
    tl.store(Y+offsets,x.to(tl.bfloat16),offsets<WORLD*CHUNK)

