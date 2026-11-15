"""Allocation is host/driver work; initialization and copies are Triton.

Storage is allocated without ATen's implicit CUDA deterministic-fill kernel.
Public empty preserves deterministic initialization: NaN for floats, integer
maximum for integers. zeros writes zeros once. No global PyTorch flag changes.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _runtime_fill(Y,N:tl.constexpr,VALUE:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    tl.store(Y+i,VALUE,i<N)


@triton.jit
def _runtime_copy(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    tl.store(Y+i,tl.load(X+i,i<N,0),i<N)


def raw_empty(shape,*,device,dtype):
    shape=tuple(shape)
    if any(type(n) is not int or n<0 for n in shape):raise ValueError('Concrete nonnegative shape required')
    device=torch.device(device)
    if device.type!='cuda':raise ValueError('CUDA runtime allocations only')
    if dtype not in (torch.float32,torch.float64,torch.float16,torch.bfloat16,torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool):
        raise TypeError('Explicit supported dtype required')
    storage=torch.UntypedStorage(math.prod(shape)*dtype.itemsize,device=device)
    return torch.empty(0,device=device,dtype=dtype).set_(storage,0,shape)


def fill_(tensor,value):
    if not tensor.is_cuda or not tensor.is_contiguous():raise ValueError('Contiguous CUDA fill required')
    if tensor.numel():_runtime_fill[(triton.cdiv(tensor.numel(),1024),)](tensor,tensor.numel(),value,1024)
    return tensor


def empty(shape,*,device,dtype):
    result=raw_empty(shape,device=device,dtype=dtype)
    if torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory:
        value=float('nan') if dtype.is_floating_point else (True if dtype==torch.bool else torch.iinfo(dtype).max)
        fill_(result,value)
    return result


def zeros(shape,*,device,dtype):
    return fill_(raw_empty(shape,device=device,dtype=dtype),0)


def copy_(target,source):
    if not target.is_cuda or not source.is_cuda or target.device!=source.device:
        raise ValueError('Local same-device CUDA copy required')
    if not target.is_contiguous() or not source.is_contiguous() or target.shape!=source.shape:
        raise ValueError('Matching contiguous shapes required')
    if target.numel():_runtime_copy[(triton.cdiv(target.numel(),1024),)](source,target,target.numel(),1024)
    return target
