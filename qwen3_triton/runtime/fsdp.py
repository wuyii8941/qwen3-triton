"""FSDP host orchestration with explicit Triton AG/RS data channels.

Custom Comm is a supported interface in pinned Torch2.10. Python FSDP hooks,
DTensor layout metadata and CUDA stream/event management remain host/runtime
services. Packing, casting, copies and accumulation require TritonRuntimeMode;
installing this adapter alone is not a complete Triton implementation.
"""
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather,ReduceScatter
from .memory import empty


class _Allocation:
    def allocate(self,size,*,dtype,device):return empty(tuple(size),device=device,dtype=dtype)


class TritonAllGather(_Allocation,AllGather):
    def __init__(self,communication):self.communication=communication;self.calls=0
    def __call__(self,output_tensor,input_tensor,group,async_op=False):
        if async_op:raise ValueError('Stream/event ordered FSDP calls required')
        if dist.get_process_group_ranks(group)!=[0,1,2,3]:raise ValueError('Canonical group required')
        self.communication.all_gather(output_tensor,input_tensor);self.calls+=1


class TritonReduceScatter(_Allocation,ReduceScatter):
    def __init__(self,communication):self.communication=communication;self.calls=0
    def __call__(self,output_tensor,input_tensor,group,op,async_op=False):
        if async_op:raise ValueError('Stream/event ordered FSDP calls required')
        if dist.get_process_group_ranks(group)!=[0,1,2,3]:raise ValueError('Canonical group required')
        if op==dist.ReduceOp.AVG:operation='avg'
        elif op==dist.ReduceOp.SUM:operation='sum'
        else:raise ValueError('FSDP SUM/AVG only')
        self.communication.reduce_scatter(output_tensor,input_tensor,operation,lossless_bf16=True);self.calls+=1


def configure(model,communication):
    if torch.__version__.split('+')[0]!='2.10.0':raise ValueError('Pinned Torch2.10 required')
    ag=TritonAllGather(communication);rs=TritonReduceScatter(communication)
    count=0
    for module in model.modules():
        if isinstance(module,FSDPModule):
            module.set_custom_all_gather(ag);module.set_custom_reduce_scatter(rs);count+=1
    if not count:raise ValueError('No FSDP units to configure')
    return dict(all_gather=ag,reduce_scatter=rs,units=count)
