"""Optional FSDP adapter for checked lossless BF16 peer staging + FP32 RS.

FP32 in/out buffer types and original SUM/AVG normalization are preserved.
Every staged coordinate on every call is checked for a lossless BF16 roundtrip.
The peer kernel changes FP32 reduction order. An external-memory reserve lowers
the Torch allocator limit, so new peer storage does not bypass the old budget.
Not qualified until complete-model gradients, quality, timing and traces pass.
"""
import os,subprocess
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatter
from symmetric_rs_v13 import PeerReduceScatter,check_wire_guard

EXTERNAL_RESERVE=512*2**20
CAPACITY=192946432


def own_nvml_mib():
    data=subprocess.check_output(['nvidia-smi','-i',str(torch.cuda.current_device()),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    values=[int(fields[1]) for line in data.splitlines() if len(fields:=[x.strip() for x in line.split(',')])==2 and int(fields[0])==os.getpid()]
    if len(values)!=1:raise RuntimeError('Need one authoritative own-process NVML memory row')
    return values[0]


class FSDPPeerReduceScatter(ReduceScatter):
    def __init__(self):
        if torch.__version__.split('+')[0]!='2.10.0' or dist.get_world_size()!=4:
            raise ValueError('Audited Torch 2.10/four-rank layout required')
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'0,1,2,3'):
            raise ValueError('NVML accounting requires identity GPU mapping 0,1,2,3')
        self.guard_selftest_passed=check_wire_guard()
        torch.cuda.set_per_process_memory_fraction((41.875*2**30-EXTERNAL_RESERVE)/torch.cuda.get_device_properties().total_memory)
        torch.cuda.synchronize();before=own_nvml_mib()
        self.peer=PeerReduceScatter(CAPACITY,torch.bfloat16)
        torch.cuda.synchronize();after=own_nvml_mib()
        observed=max(0,after-before)*2**20
        size=torch.tensor(observed,device='cuda',dtype=torch.int64);dist.all_reduce(size,op=dist.ReduceOp.MAX)
        self.observed_extra_bytes=int(size)
        # Also leave a large margin over the tensor plus its signal pad for
        # VMM granularity and CUDA bookkeeping. Record both conservative
        # reserved budget and observed process-memory delta.
        if self.observed_extra_bytes>EXTERNAL_RESERVE or self.peer.handle.buffer_size+self.peer.handle.signal_pad_size>EXTERNAL_RESERVE:
            raise RuntimeError('Peer allocation exceeds its external-memory reserve')
        self.native_calls=0;self.peer_calls=0

    def allocate(self,size,*,dtype,device):
        return torch.empty(tuple(size),dtype=dtype,device=device)

    def __call__(self,output_tensor,input_tensor,group,op,async_op=False):
        if dist.get_process_group_ranks(group)!=[0,1,2,3]:raise ValueError('Canonical four-rank order required')
        if input_tensor.dtype!=torch.float32 or output_tensor.dtype!=torch.float32:raise TypeError('FP32 RS interface required')
        n=input_tensor.numel()
        if n*4<64*2**20 or n>CAPACITY:
            self.native_calls+=1
            return dist.reduce_scatter_tensor(output_tensor,input_tensor,op=op,group=group,async_op=async_op)
        if async_op:raise ValueError('FSDP peer RS supports stream-ordered synchronous calls only')
        self.peer(output_tensor,input_tensor,op);self.peer_calls+=1

    def report(self):
        report=self.peer.report()
        if not report['lossless']:raise RuntimeError('A staged FP32 value was not exactly representable in BF16')
        report.update(mode='peer_bf16_checked',native_calls=self.native_calls,peer_calls=self.peer_calls,
            guard_positive_negative_selftest_passed=self.guard_selftest_passed,
            external_memory_budget_bytes=EXTERNAL_RESERVE,observed_extra_process_bytes=self.observed_extra_bytes,
            torch_allocator_cap_GiB=41.375,total_effective_cap_GiB=41.875,
            external_memory_not_in_PyTorch_allocator_budget_yet=False,
            external_storage_accounted_in_effective_budget=True)
        return report


def configure(model):
    # Zeroing gradients each update is separately kept identical to V12.
    # Any violation of the expected already-rounded source values is also
    # detected dynamically, rather than inferred only from this setup.
    comm=FSDPPeerReduceScatter()
    for module in model.modules():
        if isinstance(module,FSDPModule):module.set_custom_reduce_scatter(comm)
    return comm
