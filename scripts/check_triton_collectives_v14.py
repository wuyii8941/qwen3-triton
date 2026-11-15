"""Independent CPU oracles for every coordinate of real four-GPU exchanges.

Tests tail chunks, dtypes/operations, data changes, independent AG/RS streams,
sticky lossless rejection, system barrier timeout, actual decoder-sized RS,
and a candidate-only profile. Reference computation stays on CPU; GPU runtime
allocation/rendezvous uses CUDA+Gloo, never a native NCCL reference collective.
"""
import argparse,gc,hashlib,json,os
from collections import Counter
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from qwen3_triton.runtime.collectives import TritonCollectives


def values(rank,n,tag,dtype):
    x=(torch.arange(n,dtype=torch.int32)*3+rank*17+tag*11).remainder(257)-128
    return x.to(dtype)/32 if dtype.is_floating_point else x.to(dtype)


def verify(actual,reference,*,exact):
    got=actual.cpu();assert got.shape==reference.shape and got.dtype==reference.dtype
    if exact:passed=torch.equal(got,reference)
    else:
        rtol,atol=(1e-12,1e-12) if got.dtype==torch.float64 else (2e-6,1e-7)
        passed=torch.allclose(got,reference,rtol=rtol,atol=atol)
    delta=(got.double()-reference.double()).abs()
    result=dict(passed=passed,max_abs=float(delta.max()),bitwise_equal=torch.equal(got,reference),coordinates=got.numel())
    assert passed,result
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
    dist.init_process_group('gloo',timeout=timedelta(minutes=20))
    try:
        torch.cuda.synchronize()
        setup=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if rank==0 else None
        if setup is not None:setup.__enter__()
        comm=TritonCollectives(capacities=dict.fromkeys(['gather','scatter','reduce','alltoall'],4096))
        if setup is not None:
            setup.__exit__(None,None,None);path=a.output/'collectives_setup_trace.json';setup.export_chrome_trace(str(path))
            events=json.loads(path.read_text())['traceEvents'];counts=Counter(e['name'] for e in events if e.get('cat')=='kernel')
            assert not set(counts)-{'_runtime_fill'} and counts['_runtime_fill']>0,counts
            assert not [e for e in events if e.get('cat') in ['gpu_memset','gpu_memcpy']], 'Native GPU initialization/data copy during peer bootstrap'
            setup_trace=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kernels=dict(counts),only_Triton=True,no_native_GPU_memset_or_copy=True)
        else:setup_trace=None
        dist.barrier()
        cases=[]
        for dtype in [torch.bfloat16,torch.float32,torch.float64,torch.int64,torch.int32]:
            for n in [1,7,1031,4099]:
                host=values(rank,n,3,dtype);source=host.cuda();output=torch.empty(4*n,device='cuda',dtype=dtype)
                comm.all_gather(output,source)
                cases.append(dict(kind='gather',dtype=str(dtype),n=n,check=verify(output,torch.cat([values(r,n,3,dtype) for r in range(4)]),exact=True)))
                host=values(rank,4*n,7,dtype);source=host.cuda();output=torch.empty_like(source)
                comm.all_to_all(output,source)
                expected=torch.cat([values(r,4*n,7,dtype)[rank*n:(rank+1)*n] for r in range(4)])
                cases.append(dict(kind='alltoall',dtype=str(dtype),n=n,check=verify(output,expected,exact=True)))
                if dtype!=torch.bfloat16:
                    for op in ['sum','avg','max','min'] if dtype.is_floating_point else ['sum','max','min']:
                        host=values(rank,n,13,dtype);source=host.cuda();stack=torch.stack([values(r,n,13,dtype) for r in range(4)])
                        expected={'sum':lambda:stack.sum(0),'avg':lambda:stack.mean(0),'max':lambda:stack.max(0).values,'min':lambda:stack.min(0).values}[op]()
                        # Torch promotes int32 CPU sum; this protocol's output
                        # retains dtype and these small values cannot overflow.
                        expected=expected.to(dtype);comm.all_reduce(source,op)
                        cases.append(dict(kind='reduce',op=op,dtype=str(dtype),n=n,check=verify(source,expected,exact=not dtype.is_floating_point)))
        for n in [1,7,1031,4099]:
            for dtype in [torch.bfloat16,torch.float32]:
                for op in ['sum','avg']:
                    host=values(rank,4*n,19,dtype);source=host.cuda();output=torch.empty(n,device='cuda',dtype=torch.float32)
                    stack=torch.stack([values(r,4*n,19,dtype)[rank*n:(rank+1)*n].double() for r in range(4)])
                    expected=(stack.sum(0)/(4 if op=='avg' else 1)).float()
                    comm.reduce_scatter(output,source,op,lossless_bf16=dtype==torch.float32)
                    cases.append(dict(kind='scatter',op=op,dtype=str(dtype),n=n,check=verify(output,expected,exact=True)))
        # Repeated changed inputs expose stale peer cache or signal reuse.
        repeats=[]
        for tag in range(32):
            n=67;source=values(rank,4*n,tag,torch.float32).cuda();output=torch.empty(n,device='cuda')
            comm.reduce_scatter(output,source,'avg',lossless_bf16=True)
            expected=sum(values(r,4*n,tag,torch.float32)[rank*n:(rank+1)*n].double() for r in range(4)).div(4).float()
            repeats.append(verify(output,expected,exact=True))
        # Cross-rank host enqueue order differs for independent AG/RS streams.
        # Channels must not share a payload or a signal sequence.
        n=33;sg=values(rank,n,31,torch.float32).cuda();sr=values(rank,4*n,37,torch.float32).cuda()
        og=torch.empty(4*n,device='cuda');orr=torch.empty(n,device='cuda')
        comm.all_gather(og,sg);comm.reduce_scatter(orr,sr,'avg',lossless_bf16=True);torch.cuda.synchronize();dist.barrier()
        s0=torch.cuda.Stream();s1=torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream());s1.wait_stream(torch.cuda.current_stream())
        def gather():
            with torch.cuda.stream(s0):comm.all_gather(og,sg)
        def scatter():
            with torch.cuda.stream(s1):comm.reduce_scatter(orr,sr,'avg',lossless_bf16=True)
        if rank%2:scatter();gather()
        else:gather();scatter()
        torch.cuda.synchronize()
        stream_checks=[verify(og,torch.cat([values(r,n,31,torch.float32) for r in range(4)]),exact=True),
            verify(orr,sum(values(r,4*n,37,torch.float32)[rank*n:(rank+1)*n].double() for r in range(4)).div(4).float(),exact=True)]
        # A separate candidate-only profile contains no native GPU collectives,
        # CUDA barriers or arithmetic. Allocation/checks stay outside capture.
        ar=values(rank,n,41,torch.float64).cuda();at=values(rank,4*n,43,torch.int64).cuda();ato=torch.empty_like(at)
        profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if rank==0 else None
        if profile is not None:profile.__enter__()
        comm.all_gather(og,sg);comm.reduce_scatter(orr,sr,'avg',lossless_bf16=True);comm.all_reduce(ar);comm.all_to_all(ato,at)
        torch.cuda.synchronize()
        if profile is not None:
            profile.__exit__(None,None,None);path=a.output/'collectives_trace.json';profile.export_chrome_trace(str(path))
            counts=Counter(e['name'] for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel')
            assert counts=={'_runtime_stage':4,'_runtime_peer_barrier':8,'_runtime_peer_collective':4},counts
            trace=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kernels=dict(counts),only_Triton=True)
        else:trace=None
        positive=comm.report()
        # Negative flag on one rank must persist after a later exact transport.
        bad=torch.full((4,),.1 if rank==0 else .5,device='cuda');one=torch.empty(1,device='cuda')
        comm.reduce_scatter(one,bad,'avg',lossless_bf16=True)
        rejected=False
        try:comm.report()
        except RuntimeError:rejected=True
        assert rejected==(rank==0)
        good=torch.full((4,),.5,device='cuda');comm.reduce_scatter(one,good,'avg',lossless_bf16=True)
        sticky=False
        try:comm.report()
        except RuntimeError:sticky=True
        assert sticky==(rank==0)
        # Bounded failure of an incomplete barrier, without trapping/resetting
        # CUDA or clearing a live error flag. Do not reuse this test object.
        dist.barrier()
        if rank==0:
            channel=comm.channels['gather'];channel.timeout_ms=100;channel.barrier();torch.cuda.synchronize()
        dist.barrier();timeout_flags=comm.error.cpu().tolist()
        assert bool(timeout_flags[0])==(rank==0)
        comm.close();del comm;gc.collect();torch.cuda.synchronize();dist.barrier()
        # Actual full decoder bucket, checked at every coordinate. The integer
        # construction gives exact BF16 inputs and an independent FP64 oracle.
        large=TritonCollectives(capacities=dict(gather=4096,scatter=370*2**20,reduce=4096,alltoall=4096))
        total=192946432;q=total//4;source=values(rank,total,53,torch.float32).cuda()
        output=torch.empty(q,device='cuda');large.reduce_scatter(output,source,'avg',lossless_bf16=True)
        idx=torch.arange(rank*q,(rank+1)*q,dtype=torch.int32);expected=torch.zeros(q,dtype=torch.float64)
        for r in range(4):expected+=((idx*3+r*17+53*11).remainder(257)-128).double()/128
        full_check=verify(output,expected.float(),exact=True);large_report=large.report()
        report=dict(rank=rank,complete=True,passed=True,cases=cases,changed_input_repeats=repeats,stream_checks=stream_checks,
            positive_communication=positive,nonrepresentable_rejected_on_owner=rejected,sticky_guard_preserved=sticky,
            incomplete_barrier_timeout_flag=timeout_flags[0],negative_cases_as_expected=True,
            actual_decoder_elements=total,actual_decoder_FP64_oracle=full_check,actual_decoder_report=large_report,candidate_only_trace=trace,setup_trace=setup_trace,scope=__doc__)
        large.close()
        reports=[None]*4;dist.all_gather_object(reports,report)
        if rank==0:(a.output/'check.json').write_text(json.dumps(dict(complete=True,passed=True,kind='collectives',ranks=reports),indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
