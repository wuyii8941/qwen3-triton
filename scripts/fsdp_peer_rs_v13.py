"""Optional FSDP adapter for checked lossless BF16 peer staging + FP32 RS.

FP32 in/out buffer types and original SUM/AVG normalization are preserved.
Every staged coordinate on every call is checked for a lossless BF16 roundtrip.
The peer kernel changes FP32 reduction order. An external-memory reserve lowers
the Torch allocator limit, so new peer storage does not bypass the old budget.
Not qualified until complete-model gradients, quality, timing and traces pass.
"""
from contextlib import contextmanager
import gc,os,subprocess
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
        self.error=torch.zeros((),device='cuda',dtype=torch.int32)
        self.peer=None;self.native_calls=0;self.peer_calls=0
        self.completed_calls=0;self.completed_elements=0
        self.pool_generations=0;self.observed_extra_bytes=0;self.validation_cycles=[]
        self._activate_peer()

    def _activate_peer(self):
        if self.peer is not None:raise RuntimeError('Peer buffer is already active')
        if int(self.error):raise RuntimeError('Lossless violation persists across validation')
        torch.cuda.synchronize();torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction((41.875*2**30-EXTERNAL_RESERVE)/torch.cuda.get_device_properties().total_memory)
        if torch.cuda.memory_reserved()>41.375*2**30:raise RuntimeError('Live Torch storage leaves no external-buffer budget')
        torch.cuda.synchronize();before=own_nvml_mib()
        self.peer=PeerReduceScatter(CAPACITY,torch.bfloat16,error_flag=self.error)
        torch.cuda.synchronize();after=own_nvml_mib()
        observed=max(0,after-before)*2**20
        size=torch.tensor(observed,device='cuda',dtype=torch.int64);dist.all_reduce(size,op=dist.ReduceOp.MAX)
        self.observed_extra_bytes=max(self.observed_extra_bytes,int(size))
        # Also leave a large margin over the tensor plus its signal pad for
        # VMM granularity and CUDA bookkeeping. Record both conservative
        # reserved budget and observed process-memory delta.
        if self.observed_extra_bytes>EXTERNAL_RESERVE or self.peer.handle.buffer_size+self.peer.handle.signal_pad_size>EXTERNAL_RESERVE:
            raise RuntimeError('Peer allocation exceeds its external-memory reserve')
        if self.peer.error.data_ptr()!=self.error.data_ptr():raise RuntimeError('Sticky lossless flag was replaced')
        self.pool_generations+=1

    @contextmanager
    def validation_memory(self):
        if self.peer is None:raise RuntimeError('Nested peer validation context')
        torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
        snapshot=self.peer.report()
        if not snapshot['lossless']:raise RuntimeError('Lossless violation before validation')
        before=own_nvml_mib();torch_before=torch.cuda.memory_reserved()
        self.completed_calls+=snapshot['calls'];self.completed_elements+=snapshot['checked_elements']
        self.peer=None
        gc.collect();torch.cuda.synchronize()
        after=own_nvml_mib();torch_after=torch.cuda.memory_reserved()
        # Correct for any simultaneous Torch cache release; only count the
        # observed disappearance of external storage. NVML reports whole MiB.
        external_released=(before-after)*2**20-(torch_before-torch_after)
        if external_released<snapshot['requested_external_buffer_bytes']-2*2**20:
            raise RuntimeError('External peer buffer release not demonstrated by NVML')
        dist.barrier();torch.cuda.synchronize()
        torch.cuda.set_per_process_memory_fraction(41.875*2**30/torch.cuda.get_device_properties().total_memory)
        event=dict(cycle=len(self.validation_cycles),peer_calls_before=self.peer_calls,
            peer_buffer_released_before_validation=True,observed_external_release_bytes=external_released,
            requested_external_buffer_bytes=snapshot['requested_external_buffer_bytes'],
            validation_torch_allocator_cap_GiB=41.875,peer_buffers_active_during_validation=False,
            sticky_lossless_counter_unchanged=True,restored_before_training=False)
        self.validation_cycles.append(event)
        try:
            yield
        except BaseException:
            event['validation_failed']=True
            raise
        else:
            torch.cuda.synchronize()
            event.update(validation_max_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                validation_max_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
            if max(event['validation_max_allocated_GiB'],event['validation_max_reserved_GiB'])>41.875:
                raise RuntimeError('Validation exceeded the original allocator budget')
            self._activate_peer()
            event.update(restored_before_training=True,restored_pool_generation=self.pool_generations,
                training_torch_allocator_cap_GiB=41.375)

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
        if self.peer is None:raise RuntimeError('Peer gradient requested during validation')
        self.peer(output_tensor,input_tensor,op);self.peer_calls+=1

    def report(self):
        report=self.peer.report()
        if not report['lossless']:raise RuntimeError('A staged FP32 value was not exactly representable in BF16')
        report['calls']+=self.completed_calls;report['checked_elements']+=self.completed_elements
        if report['calls']!=self.peer_calls:raise RuntimeError('Peer call count lost across validation')
        report.update(mode='peer_bf16_checked',native_calls=self.native_calls,peer_calls=self.peer_calls,
            guard_positive_negative_selftest_passed=self.guard_selftest_passed,
            external_memory_budget_bytes=EXTERNAL_RESERVE,observed_extra_process_bytes=self.observed_extra_bytes,
            torch_allocator_cap_GiB=41.375,total_effective_cap_GiB=41.875,
            external_memory_not_in_PyTorch_allocator_budget_yet=False,
            external_storage_accounted_in_effective_budget=True,pool_generations=self.pool_generations,
            validation_cycles=[dict(event) for event in self.validation_cycles],sticky_lossless_flag_persisted_across_validation=True)
        return report


def configure(model):
    # Zeroing gradients each update is separately kept identical to V12.
    # Any violation of the expected already-rounded source values is also
    # detected dynamically, rather than inferred only from this setup.
    comm=FSDPPeerReduceScatter()
    for module in model.modules():
        if isinstance(module,FSDPModule):module.set_custom_reduce_scatter(comm)
    model._peer_gradient_communication=comm
    return comm
