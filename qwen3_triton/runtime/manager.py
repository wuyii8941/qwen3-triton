"""Explicit collective selection, peer-memory accounting and coordination."""
from contextlib import contextmanager
import os,subprocess
import torch
import torch.distributed as dist
from .memory import zeros
from .dispatch import TritonRuntimeMode
from .collectives import TritonCollectives
from .hybrid_collectives import HybridCollectives
from .fsdp import configure

CAP_GIB=41.875
RESERVE=512*2**20


def own_nvml_mib():
    rows=subprocess.check_output(['nvidia-smi','-i',str(torch.cuda.current_device()),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    values=[int(f[1]) for line in rows.splitlines() if len(f:=[x.strip() for x in line.split(',')])==2 and int(f[0])==os.getpid()]
    if len(values)!=1:raise RuntimeError('Own-process NVML row unavailable')
    return values[0]


class TrainingRuntime:
    def __init__(self,kind,cpu_group):
        if kind not in ('full','hybrid'):raise ValueError('Explicit full or hybrid collective mode required')
        self.kind=kind;self.cpu_group=cpu_group;self.full=kind=='full'
        self.mode=TritonRuntimeMode()
        self.communication=None;self.hooks=None;self.observed_extra_bytes=0
        self.validation_cycles=[];self.phase_peaks=[]
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'0,1,2,3'):raise ValueError('Identity device mapping required for accounting')
        warm=zeros((1,),device='cuda',dtype=torch.int32);torch.cuda.synchronize();del warm;torch.cuda.empty_cache()
        total=torch.cuda.get_device_properties().total_memory
        torch.cuda.set_per_process_memory_fraction((CAP_GIB*2**30-RESERVE)/total)
        before=own_nvml_mib();self.peer=TritonCollectives()
        if kind=='hybrid':
            self.nccl_group=dist.new_group(ranks=[0,1,2,3],backend='nccl')
            self.communication=HybridCollectives(self.peer,self.nccl_group)
        else:
            self.nccl_group=None;self.communication=self.peer
        torch.cuda.synchronize()
        extra=max(0,own_nvml_mib()-before)*2**20
        observed=[None]*4;dist.all_gather_object(observed,extra,group=cpu_group)
        self.observed_extra_bytes=max(observed)
        report=self.peer.report()
        mapped=sum(c['mapped_buffer_and_signal_bytes'] for c in report['channels'].values())
        if self.observed_extra_bytes>RESERVE or mapped>RESERVE:raise RuntimeError('External peer buffers exceed conservative512MiB budget')

    def scope(self):return self.mode

    def bind(self,model):
        self.hooks=configure(model,self.communication)
        if hasattr(model.lm_head,'communication'):model.lm_head.communication=self.communication

    def reduce(self,tensor,op='sum'):
        self.communication.all_reduce(tensor,op)
        return tensor

    def barrier(self):
        self.communication.barrier()

    @contextmanager
    def validation_memory(self):
        torch.cuda.synchronize();torch.cuda.empty_cache()
        yield
        torch.cuda.synchronize()
        self.validation_cycles.append(dict(peer_buffers_needed_for_validation_all_gather=True,
            torch_allocator_cap_GiB=CAP_GIB-RESERVE/2**30,
            original_effective_cap_GiB=CAP_GIB,max_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            max_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))

    @property
    def extra_bytes(self):return RESERVE

    def record_peak(self,phase,*,external_bytes=None,reset=False):
        extra=self.extra_bytes if external_bytes is None else external_bytes
        peak=dict(allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        row=dict(phase=phase,torch=peak,external_budget_bytes=extra,effective={k:v+extra/2**30 for k,v in peak.items()})
        self.phase_peaks.append(row)
        if reset:torch.cuda.reset_peak_memory_stats()
        return row

    def report(self):
        comm=self.communication.report() if self.communication is not None else dict(mode='native')
        comm.update(external_memory_budget_bytes=RESERVE,observed_extra_process_bytes=self.observed_extra_bytes,
            torch_allocator_cap_GiB=CAP_GIB-RESERVE/2**30,total_effective_cap_GiB=CAP_GIB,
            FSDP_host_units=self.hooks['units'],FSDP_AG_calls=self.hooks['all_gather'].calls,FSDP_RS_calls=self.hooks['reduce_scatter'].calls,
            validation_cycles=self.validation_cycles)
        return dict(kind=self.kind,communication=comm,phase_memory_ledger=self.phase_peaks,strict_GPU_dispatch=self.mode.report() if self.mode is not None else None,
            CPU_bootstrap_and_metadata_backend=dist.get_backend(self.cpu_group),
            host_services='FSDP/autograd metadata, CUDA allocator/peer mappings/streams/events, CPU file IO and host-device DMA.',
            collective_scope='Every GPU collective is Triton.' if self.full else 'Large FP32 reduce-scatter is Triton; remaining collectives are NCCL.')
