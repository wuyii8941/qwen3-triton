"""Four local GPU collectives whose data and synchronization kernels are Triton.

CUDA Symmetric Memory provides allocation, peer mapping and CPU rendezvous.
No native NCCL/CUDA collective or CUDA barrier kernel is called. Separate
channels preserve FSDP AG/RS stream ordering without aliasing shared staging.
System-scope release stores/acquire loads use one writer per mailbox, not
unsupported remote read-modify-write atomics. Hardware correctness must be
established by the real multi-rank checks before model integration/adoption.
"""
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
import triton.language as tl
from .memory import zeros,fill_


@triton.jit
def _sys_load(P):
    return tl.inline_asm_elementwise('ld.acquire.sys.global.u32 $0, [$1];',constraints='=r,l',args=[P],dtype=tl.uint32,is_pure=False,pack=1)


@triton.jit
def _timer():
    return tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;',constraints='=l',args=[],dtype=tl.uint64,is_pure=False,pack=1)


@triton.jit(do_not_specialize=['EPOCH'])
def _runtime_peer_barrier(F0,F1,F2,F3,LOCAL,ERROR,EPOCH,RANK:tl.constexpr,TIMEOUT_NS:tl.constexpr):
    lane=tl.arange(0,32)
    target=tl.where(lane==0,F0,tl.where(lane==1,F1,tl.where(lane==2,F2,F3)))+RANK*32
    tl.inline_asm_elementwise('''{
        .reg .pred active;
        setp.ne.u32 active, $3, 0;
        @active st.release.sys.global.u32 [$1], $2;
        mov.u32 $0, 0;
    }''',constraints='=r,l,r,r',args=[target,EPOCH.to(tl.uint32),(lane<4).to(tl.uint32)],dtype=tl.uint32,is_pure=False,pack=1)
    address=LOCAL+tl.minimum(lane,3)*32
    started=_timer();seen=_sys_load(address);expired=tl.full((),False,tl.int1)
    while (tl.sum(((lane<4)&(seen<EPOCH)).to(tl.int32),0)>0)&(~expired):
        expired=_timer()-started>TIMEOUT_NS
        seen=_sys_load(address)
    if expired:tl.atomic_or(ERROR,1)


@triton.jit
def _runtime_stage(SOURCE,WIRE,ERROR,Q:tl.constexpr,START:tl.constexpr,CHUNK:tl.constexpr,
                   PACK4:tl.constexpr,CHECK:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);n=CHUNK*(4 if PACK4 else 1);valid=i<n
    source=(i//CHUNK)*Q+START+i%CHUNK if PACK4 else START+i
    value=tl.load(SOURCE+source,valid,0);wire=value.to(WIRE.dtype.element_ty)
    tl.store(WIRE+i,wire,valid)
    if CHECK:
        restored=wire.to(tl.float32)
        bad=tl.sum((valid&(value.to(tl.int32,bitcast=True)!=restored.to(tl.int32,bitcast=True))).to(tl.int32),0)
        if bad:tl.atomic_or(ERROR+1,1)


@triton.jit
def _runtime_peer_collective(P0,P1,P2,P3,OUT,Q:tl.constexpr,START:tl.constexpr,CHUNK:tl.constexpr,
                             RANK:tl.constexpr,KIND:tl.constexpr,OP:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    if KIND=='gather' or KIND=='alltoall':
        valid=i<4*CHUNK;owner=i//CHUNK;within=i%CHUNK
        ptr=tl.where(owner==0,P0,tl.where(owner==1,P1,tl.where(owner==2,P2,P3)))
        source=within if KIND=='gather' else RANK*CHUNK+within
        value=tl.load(ptr+source,valid,0,volatile=True)
        tl.store(OUT+owner*Q+START+within,value,valid)
    else:
        valid=i<CHUNK;source=RANK*CHUNK+i if KIND=='scatter' else i
        dtype=OUT.dtype.element_ty
        a=tl.load(P0+source,valid,0,volatile=True).to(dtype)
        b=tl.load(P1+source,valid,0,volatile=True).to(dtype)
        c=tl.load(P2+source,valid,0,volatile=True).to(dtype)
        d=tl.load(P3+source,valid,0,volatile=True).to(dtype)
        if OP=='avg':a=a*.25;b=b*.25;c=c*.25;d=d*.25
        if OP=='max':value=tl.maximum(tl.maximum(a,b),tl.maximum(c,d))
        elif OP=='min':value=tl.minimum(tl.minimum(a,b),tl.minimum(c,d))
        else:value=((a+b)+c)+d
        tl.store(OUT+START+i,value,valid)


class _Channel:
    def __init__(self,name,capacity,group,error,timeout_ms):
        if capacity<4096 or capacity%16:raise ValueError('Aligned staging capacity required')
        self.name=name;self.capacity=capacity;self.rank=dist.get_rank(group);self.group=group;self.error=error
        self.epoch=0;self.barriers=0;self.calls=0;self.staged_elements=0;self.checked_elements=0;self.timeout_ms=timeout_ms
        self.payload=symm.empty(capacity,device='cuda',dtype=torch.uint8)
        self.payload_handle=symm.rendezvous(self.payload,group)
        self.flags=symm.empty(4*32,device='cuda',dtype=torch.int32)
        self.flags_handle=symm.rendezvous(self.flags,group)
        self.peers_flags=[self.flags_handle.get_buffer(r,(4*32,),torch.int32) for r in range(4)]
        self.views={};self.last_event=None
        fill_(self.flags,0);torch.cuda.synchronize();dist.barrier(group=group)

    def tensors(self,dtype):
        if dtype not in self.views:
            n=self.capacity//dtype.itemsize
            self.views[dtype]=[self.payload_handle.get_buffer(r,(n,),dtype) for r in range(4)]
        return self.views[dtype]

    def begin(self):
        if self.last_event is not None:torch.cuda.current_stream().wait_event(self.last_event)

    def barrier(self):
        self.epoch+=1
        if self.epoch>=2**31:raise RuntimeError('Signal epoch exhausted; construct a fresh communicator')
        _runtime_peer_barrier[(1,)](*self.peers_flags,self.flags,self.error,self.epoch,self.rank,self.timeout_ms*1000000,num_warps=1)
        self.barriers+=1

    def end(self):
        self.last_event=torch.cuda.Event();self.last_event.record();self.calls+=1

    def report(self):
        return dict(capacity_bytes=self.capacity,calls=self.calls,Triton_barriers=self.barriers,staged_elements=self.staged_elements,
            checked_lossless_elements=self.checked_elements,epoch=self.epoch,
            mapped_buffer_and_signal_bytes=self.payload_handle.buffer_size+self.payload_handle.signal_pad_size+self.flags_handle.buffer_size+self.flags_handle.signal_pad_size)


class TritonCollectives:
    def __init__(self,*,capacities=None,group=None,timeout_ms=30000):
        group=dist.group.WORLD if group is None else group
        if dist.get_world_size(group)!=4 or dist.get_process_group_ranks(group)!=[0,1,2,3]:raise ValueError('Four canonical local ranks required')
        if torch.__version__.split('+')[0]!='2.10.0':raise ValueError('Audited Torch2.10 CUDA symmetric API required')
        device=torch.device('cuda',torch.cuda.current_device());backend=symm.get_backend(device)
        if backend is None:symm.set_backend('CUDA')
        elif backend!='CUDA':raise RuntimeError('CUDA symmetric backend required')
        self.group=group;self.error=zeros((2,),device=device,dtype=torch.int32)
        if capacities is None:capacities=dict(gather=96*2**20,scatter=370*2**20,reduce=1*2**20,alltoall=8*2**20)
        if set(capacities)!={'gather','scatter','reduce','alltoall'}:raise ValueError('All four communication channels required')
        self.channels={name:_Channel(name,capacities[name],group,self.error,timeout_ms) for name in ['gather','scatter','reduce','alltoall']}

    def _run(self,kind,output,source,op='sum',lossless_bf16=False):
        if (not source.is_cuda or source.device!=output.device or not source.is_contiguous() or not output.is_contiguous()
            or source.dtype not in [torch.bfloat16,torch.float32,torch.float64,torch.int64,torch.int32]):
            raise ValueError('Supported contiguous CUDA tensors required')
        if kind=='scatter':
            if source.numel()!=4*output.numel() or output.dtype!=torch.float32:raise ValueError('Four-shard FP32 reduction output required')
            q=output.numel();pack=True
        elif kind=='alltoall':
            if source.numel()!=output.numel() or source.numel()%4 or source.dtype!=output.dtype:raise ValueError('Equal all-to-all blocks required')
            q=source.numel()//4;pack=True
        elif kind=='gather':
            if output.numel()!=4*source.numel() or output.dtype!=source.dtype:raise ValueError('Four-shard gather output required')
            q=source.numel();pack=False
        else:
            if output.shape!=source.shape or output.dtype!=source.dtype:raise ValueError('Matching all-reduce tensor required')
            q=source.numel();pack=False
        if op not in ['sum','avg','max','min'] or (op=='avg' and not output.dtype.is_floating_point):raise ValueError('Supported reduction required')
        if lossless_bf16 and (kind!='scatter' or source.dtype!=torch.float32):raise ValueError('Lossless BF16 staging is only for FP32 FSDP RS input')
        wire_dtype=torch.bfloat16 if lossless_bf16 else source.dtype
        channel=self.channels[kind];peers=channel.tensors(wire_dtype);wire=peers[dist.get_rank(self.group)]
        max_chunk=channel.capacity//wire_dtype.itemsize//(4 if pack else 1)
        channel.begin()
        for start in range(0,q,max_chunk):
            chunk=min(q-start,max_chunk);n=chunk*(4 if pack else 1)
            _runtime_stage[(triton.cdiv(n,1024),)](source,wire,self.error,q,start,chunk,pack,lossless_bf16,1024)
            channel.barrier()
            count=chunk*(4 if kind in ['gather','alltoall'] else 1)
            _runtime_peer_collective[(triton.cdiv(count,1024),)](*peers,output,q,start,chunk,dist.get_rank(self.group),kind,op,1024,enable_fp_fusion=False)
            channel.barrier();channel.staged_elements+=n
            if lossless_bf16:channel.checked_elements+=n
        channel.end()
        return output

    def all_gather(self,output,source):return self._run('gather',output,source)
    def reduce_scatter(self,output,source,op='avg',*,lossless_bf16=False):return self._run('scatter',output,source,op,lossless_bf16)
    def all_reduce(self,tensor,op='sum'):return self._run('reduce',tensor,tensor,op)
    def all_to_all(self,output,source):return self._run('alltoall',output,source)

    def barrier(self):
        channel=self.channels['reduce'];channel.begin();channel.barrier();channel.end()

    def report(self):
        torch.cuda.synchronize();errors=self.error.cpu().tolist()
        if any(errors):raise RuntimeError(f'Triton communicator rejected: timeout={errors[0]}, lossless={errors[1]}')
        return dict(complete=True,bootstrap_backend=dist.get_backend(self.group),GPU_data_backend='Triton',GPU_barrier_backend='Triton system release/acquire',
            native_GPU_collective_calls=0,native_CUDA_barrier_calls=0,timeout_flag=errors[0],lossless_violation_flag=errors[1],
            channels={name:c.report() for name,c in self.channels.items()})
