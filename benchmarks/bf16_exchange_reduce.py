"""Experimental BF16 transport + FP32 local reduction, not installed in FSDP.

Only original BF16 gradient contributions are supported. Casting an arbitrary
FP32 gradient to BF16 would be lossy and is deliberately rejected. Transport
preserves BF16 bits; FP32 summation order can differ from NCCL ReduceScatter,
so numerical/model gates remain necessary. Accumulated FP32 gradients are not
implicitly supported by this experiment.
"""
import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _sum_contributions(inp,out,n_cols,WORLD:tl.constexpr,SCALE:tl.constexpr,
                       BLOCK_R:tl.constexpr,BLOCK_C:tl.constexpr):
    cols=tl.program_id(0)*BLOCK_C+tl.arange(0,BLOCK_C)
    ranks=tl.arange(0,BLOCK_R)
    values=tl.load(inp+ranks[:,None]*n_cols+cols[None,:],
                   mask=(ranks[:,None]<WORLD)&(cols[None,:]<n_cols),other=0).to(tl.float32)
    # Scale BEFORE adding, matching the safe predivide form of rank-average;
    # summing huge BF16 values first could overflow FP32 unnecessarily.
    result=tl.sum(values*SCALE,axis=0)
    tl.store(out+cols,result,mask=cols<n_cols)


def sum_received(received,world,*,average=True,backend='torch'):
    if (type(world) is not int or received.dtype!=torch.bfloat16 or not received.is_contiguous() or world<2
            or not received.numel() or received.numel()%world):
        raise ValueError('Contiguous nonempty BF16 contributions divisible by world required')
    cols=received.numel()//world;scale=1/world if average else 1.
    if backend=='torch':return (received.reshape(world,cols).float()*scale).sum(dim=0)
    if backend!='triton' or not received.is_cuda:
        raise ValueError('Triton reduction requires CUDA; use explicit torch backend on CPU')
    out=torch.empty(cols,device=received.device,dtype=torch.float32)
    _sum_contributions[(triton.cdiv(cols,1024),)](received,out,cols,world,scale,
        triton.next_power_of_2(world),1024,num_warps=4)
    return out


@torch.no_grad()
def exchange_reduce(gradient,*,group=None,average=True,backend='torch',return_received=False):
    if not dist.is_initialized():raise ValueError('Initialized process group required')
    world=dist.get_world_size(group)
    if (gradient.dtype!=torch.bfloat16 or not gradient.is_contiguous() or world<2
            or not gradient.numel() or gradient.numel()%world):
        raise ValueError('Original contiguous BF16 gradient divisible by world required; no FP32 quantization')
    flat=gradient.reshape(-1)
    received=torch.empty_like(flat)
    dist.all_to_all_single(received,flat,group=group)
    reduced=sum_received(received,world,average=average,backend=backend)
    return (reduced,received) if return_received else reduced
