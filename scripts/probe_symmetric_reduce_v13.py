"""Probe CUDA symmetric-memory peer access and a Triton FP32 reduction.

BF16 source values are loaded directly from four mapped peer buffers and
accumulated/output in FP32. This tests availability/correctness only, not a
training integration, communication compression claim or speedup.
"""
import argparse
from datetime import timedelta
import json,os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
import triton.language as tl


@triton.jit
def _peer_mean(P0,P1,P2,P3,Y,Q:tl.constexpr,RANK:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);off=RANK*Q+i;mask=i<Q
    a=tl.load(P0+off,mask,0).to(tl.float32)*.25
    b=tl.load(P1+off,mask,0).to(tl.float32)*.25
    c=tl.load(P2+off,mask,0).to(tl.float32)*.25
    d=tl.load(P3+off,mask,0).to(tl.float32)*.25
    value=((a+b)+c)+d
    tl.store(Y+i,value,mask)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(20261115+rank);torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl',timeout=timedelta(seconds=90),device_id=torch.device('cuda',torch.cuda.current_device()))
    try:
        assert dist.get_world_size()==4
        n=2**18; q=n//4; setup=None; x=hdl=None
        try:
            symm.set_backend('CUDA')
            x=symm.empty(n,device=torch.device('cuda',torch.cuda.current_device()),dtype=torch.bfloat16)
            hdl=symm.rendezvous(x,dist.group.WORLD)
            setup=dict(rank=rank,supported=True)
        except RuntimeError as exc:setup=dict(rank=rank,supported=False,error=str(exc)[:1000])
        statuses=[None]*4;dist.all_gather_object(statuses,setup)
        available=all(s['supported'] for s in statuses)
        error=None
        if available:
            x.copy_(torch.randn(n,device='cuda',dtype=torch.bfloat16))
            received=torch.empty(n,device='cuda',dtype=torch.bfloat16)
            dist.all_to_all_single(received,x)
            expected=(received.view(4,q).double().sum(0)/4).float()
            out=torch.empty(q,device='cuda',dtype=torch.float32)
            peers=[hdl.get_buffer(r,(n,),torch.bfloat16) for r in range(4)]
            hdl.barrier(timeout_ms=10000)
            _peer_mean[(triton.cdiv(q,1024),)](*peers,out,q,rank,1024,enable_fp_fusion=False)
            hdl.barrier(timeout_ms=10000);torch.cuda.synchronize()
            delta=out-expected
            error=dict(relative_l2=float(delta.norm()/expected.norm().clamp_min(1e-12)),max_abs=float(delta.abs().max()),
                finite=bool(torch.isfinite(out).all()),all_coordinates=True)
            assert error['relative_l2']<1e-6 and error['max_abs']<1e-6 and error['finite'],error
        errors=[None]*4;dist.all_gather_object(errors,error)
        if rank==0:
            report=dict(complete=True,available=available,statuses=statuses,errors=errors,full_training_qualified=False,
                torch=torch.__version__,compute_capability=torch.cuda.get_device_capability(),
                source_dtype='BF16',accumulate_output_dtype='FP32',scope=__doc__)
            with args.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
