"""Audited overwrite-only FSDP buffers, using custom Comm interfaces.

Restricted to four-rank Qwen3 with even dim0 parameter shards. The all-gather
copy-in writes the complete local input slice before NCCL reads it; NCCL fills
the entire output. RS chunk_cat writes the complete input (no padding here),
and NCCL writes the entire output. No PyTorch internals are patched.
"""
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather,ReduceScatter
from qwen3_triton.overwrite_allocation import raw_overwrite_output


class _Allocation:
    def allocate(self,size,*,dtype,device):
        self.allocations+=1
        return raw_overwrite_output(tuple(size),dtype=dtype,device=device)


class OverwriteAllGather(_Allocation,AllGather):
    def __init__(self):self.allocations=0;self.calls=0
    def __call__(self,output_tensor,input_tensor,group,async_op=False):
        if output_tensor.numel()!=4*input_tensor.numel():raise ValueError('Expected four complete shards')
        self.calls+=1
        return dist.all_gather_into_tensor(output_tensor,input_tensor,group=group,async_op=async_op)


class OverwriteReduceScatter(_Allocation,ReduceScatter):
    def __init__(self,downstream):self.downstream=downstream;self.allocations=0;self.calls=0
    def __call__(self,output_tensor,input_tensor,group,op,async_op=False):
        if input_tensor.numel()!=4*output_tensor.numel():raise ValueError('Expected four complete shards')
        self.calls+=1
        if self.downstream is not None:
            return self.downstream(output_tensor,input_tensor,group,op,async_op=async_op)
        return dist.reduce_scatter_tensor(output_tensor,input_tensor,group=group,op=op,async_op=async_op)
    def report(self):
        return dict(mode='native' if self.downstream is None else self.downstream.mode,
                    overwrite_communication_buffers=True,RS_calls=self.calls,RS_allocations=self.allocations,
                    AG_calls=self.ag.calls,AG_allocations=self.ag.allocations,
                    downstream=self.downstream.report() if self.downstream is not None else None)


def configure(model,downstream):
    if torch.__version__.split('+')[0]!='2.10.0' or dist.get_world_size()!=4:
        raise ValueError('Audited Torch 2.10/four-rank layout required')
    for name,p in model.named_parameters():
        if p.ndim<1 or p.shape[0]%4:raise ValueError('Uneven parameter shards: '+name)
    ag=OverwriteAllGather();rs=OverwriteReduceScatter(downstream);rs.ag=ag
    for module in model.modules():
        if isinstance(module,FSDPModule):
            module.set_custom_all_gather(ag);module.set_custom_reduce_scatter(rs)
    return rs
