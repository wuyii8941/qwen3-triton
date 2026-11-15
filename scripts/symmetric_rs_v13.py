"""Experimental peer-memory RS: FP32 accumulation, optionally lossless BF16 wire.

No model integration yet. The checked cast flags ANY FP32 input not represented
exactly in BF16, across every call and coordinate. A nonzero flag disqualifies
the method even if a downstream gradient tolerance would otherwise pass.
CUDA-backend symmetric buffers are externally allocated and must be included
in the memory budget before any future whole-model performance claim.
"""
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
import triton.language as tl


@triton.jit
def _checked_wire_copy(X,W,ERROR,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    value=tl.load(X+i,valid,0)
    wire=value.to(W.dtype.element_ty)
    tl.store(W+i,wire,valid)
    # Check bits as well, so signed zero/subnormal behavior cannot be hidden
    # by a floating-point comparison. No extra loss is allowed on the wire.
    restored=wire.to(tl.float32)
    bad=tl.sum((valid & (value.to(tl.int32,bitcast=True)!=restored.to(tl.int32,bitcast=True))).to(tl.int32),0)
    if bad:tl.atomic_or(ERROR,1)


@triton.jit
def _peer_reduce(P0,P1,P2,P3,OUT,Q:tl.constexpr,RANK:tl.constexpr,
                 SCALE:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);off=RANK*Q+i;valid=i<Q
    a=tl.load(P0+off,valid,0,volatile=True).to(tl.float32)*SCALE
    b=tl.load(P1+off,valid,0,volatile=True).to(tl.float32)*SCALE
    c=tl.load(P2+off,valid,0,volatile=True).to(tl.float32)*SCALE
    d=tl.load(P3+off,valid,0,volatile=True).to(tl.float32)*SCALE
    value=((a+b)+c)+d
    tl.store(OUT+i,value,valid)


class PeerReduceScatter:
    def __init__(self,capacity,wire_dtype,*,block=1024,warps=4,error_flag=None):
        if dist.get_world_size()!=4 or capacity%4:raise ValueError('Four equal ranks required')
        if wire_dtype not in (torch.bfloat16,torch.float32):raise ValueError('BF16/FP32 wire required')
        device=torch.device('cuda',torch.cuda.current_device())
        backend=symm.get_backend(device)
        if backend is None:symm.set_backend('CUDA')
        elif backend!='CUDA':raise RuntimeError('CUDA symmetric backend required')
        self.wire=symm.empty(capacity,device=device,dtype=wire_dtype)
        self.handle=symm.rendezvous(self.wire,dist.group.WORLD)
        self.peers=[self.handle.get_buffer(r,(capacity,),wire_dtype) for r in range(4)]
        if error_flag is not None and (error_flag.dtype!=torch.int32 or error_flag.numel()!=1 or error_flag.device!=device):
            raise ValueError('Shared error flag must be a local CUDA int32 scalar')
        self.error=torch.zeros((),device='cuda',dtype=torch.int32) if error_flag is None else error_flag
        self.rank=dist.get_rank();self.capacity=capacity;self.block=block;self.warps=warps
        self.calls=0;self.checked_elements=0

    def __call__(self,out,source,op=dist.ReduceOp.AVG):
        n=source.numel()
        if (n>self.capacity or n!=out.numel()*4 or source.dtype!=torch.float32 or out.dtype!=torch.float32
            or not source.is_contiguous() or not out.is_contiguous()):
            raise ValueError('Matching contiguous FP32 input/output within capacity required')
        if op not in (dist.ReduceOp.AVG,dist.ReduceOp.SUM):raise ValueError('AVG/SUM only')
        _checked_wire_copy[(triton.cdiv(n,1024),)](source,self.wire,self.error,n,1024)
        self.handle.barrier(timeout_ms=30000)
        q=n//4
        _peer_reduce[(triton.cdiv(q,self.block),)](*self.peers,out,q,self.rank,
            .25 if op==dist.ReduceOp.AVG else 1.,self.block,num_warps=self.warps,enable_fp_fusion=False)
        self.handle.barrier(timeout_ms=30000)
        self.calls+=1;self.checked_elements+=n

    def report(self):
        bad=int(self.error)
        return dict(wire_dtype=str(self.wire.dtype),accumulate_output_dtype='torch.float32',calls=self.calls,
            lossless_violation_flag=bad,lossless=bad==0,checked_elements=self.checked_elements,
            requested_external_buffer_bytes=self.capacity*self.wire.element_size(),
            handle_buffer_size=self.handle.buffer_size,signal_pad_size=self.handle.signal_pad_size,
            block=self.block,warps=self.warps,external_memory_not_in_PyTorch_allocator_budget_yet=True)


def check_wire_guard():
    """Separate flags prove acceptance/rejection without clearing live state."""
    positive=torch.tensor([0.,-0.,.5,-.5,1.,-2.,.0078125],device='cuda',dtype=torch.bfloat16).float()
    wire=torch.empty(positive.numel(),device='cuda',dtype=torch.bfloat16)
    good=torch.zeros((),device='cuda',dtype=torch.int32)
    _checked_wire_copy[(1,)](positive,wire,good,positive.numel(),1024)
    assert int(good)==0 and torch.equal(positive.view(torch.int32),wire.float().view(torch.int32))
    negative=torch.tensor([.1],device='cuda',dtype=torch.float32)
    bad=torch.zeros((),device='cuda',dtype=torch.int32)
    _checked_wire_copy[(1,)](negative,wire,bad,1,1024)
    assert int(bad)==1
    return True
