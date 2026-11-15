"""Measure checked-cast + peer FP32 RS including both GPU barriers.

BF16-valued FP32 sources model an already-rounded local gradient, without
claiming this property holds for an integrated training path. Every coordinate
is checked against routed FP64 references. Still a module-only experiment.
"""
import argparse,json,os,statistics,time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from symmetric_rs_v13 import PeerReduceScatter,_checked_wire_copy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));rank=int(os.environ['RANK'])
    torch.set_num_threads(2);torch.manual_seed(20261115+rank)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
    dist.init_process_group('nccl',timeout=timedelta(minutes=5),device_id=torch.device('cuda',torch.cuda.current_device()))
    try:
        if rank==0:a.output.mkdir(exist_ok=False)
        dist.barrier()
        # A real nonrepresentable FP32 value must be detected by the exactness
        # guard; the guard belongs to a separate test buffer, never reset on a
        # live communication instance.
        negative=torch.tensor([.1],device='cuda',dtype=torch.float32)
        temp=torch.empty(1,device='cuda',dtype=torch.bfloat16);flag=torch.zeros((),device='cuda',dtype=torch.int32)
        _checked_wire_copy[(1,)](negative,temp,flag,1,1024);assert int(flag)==1
        records=[]
        for n in [2**20,16*2**20,192946432]:
            source=torch.randn(n,device='cuda',dtype=torch.bfloat16).float()
            receive=torch.empty_like(source);dist.all_to_all_single(receive,source)
            exact=receive.view(4,n//4).double().sum(0);del receive
            out=torch.empty(n//4,device='cuda',dtype=torch.float32)
            # Reuse one external buffer per wire type; allocation/rendezvous
            # is excluded, copy and both barriers are inside each sample.
            bf16=PeerReduceScatter(n,torch.bfloat16)
            fp32=PeerReduceScatter(n,torch.float32)
            errors={}
            for op_name,op in [('AVG',dist.ReduceOp.AVG),('SUM',dist.ReduceOp.SUM)]:
                expected=(exact/4 if op_name=='AVG' else exact).float()
                for name,comm in [('native',None),('peer_bf16',bf16),('peer_fp32',fp32)]:
                    if comm is None:dist.reduce_scatter_tensor(out,source,op=op)
                    else:comm(out,source,op)
                    delta=out-expected
                    error=dict(relative_l2=float(delta.norm()/expected.norm().clamp_min(1e-12)),max_abs=float(delta.abs().max()),finite=bool(torch.isfinite(out).all()))
                    assert error['relative_l2']<1e-6 and error['max_abs']<2e-6 and error['finite'],(n,name,error)
                    errors[name+'_'+op_name]=error
                    del delta
                del expected
            del exact
            for order,modes in enumerate([['native','peer_fp32','peer_bf16'],['peer_bf16','peer_fp32','native']],1):
                for mode in modes:
                    comm=dict(peer_bf16=bf16,peer_fp32=fp32).get(mode)
                    def run():
                        if comm is None:dist.reduce_scatter_tensor(out,source,op=dist.ReduceOp.AVG)
                        else:comm(out,source)
                    for _ in range(3):run()
                    times=[]
                    for _ in range(8):
                        torch.cuda.synchronize();dist.barrier();start=time.perf_counter()
                        run();torch.cuda.synchronize()
                        duration=torch.tensor(time.perf_counter()-start,device='cuda',dtype=torch.float64)
                        dist.all_reduce(duration,op=dist.ReduceOp.MAX);times.append(float(duration))
                    row=dict(input_numel=n,order=order,mode=mode,samples=times,median=statistics.median(times),errors=errors)
                    records.append(row)
                    if rank==0:print(json.dumps(row),flush=True)
            for comm in [bf16,fp32]:assert comm.report()['lossless'],comm.report()
            records[-1]['communication_reports']=[bf16.report(),fp32.report()]
            # Repeated writes with changed values check that peer loads do not
            # return a prior call's cache contents.
            source.neg_();bf16(out,source)
            ref=torch.empty_like(out);dist.reduce_scatter_tensor(ref,source,op=dist.ReduceOp.AVG)
            assert torch.equal(out,ref) or bool(torch.allclose(out,ref,rtol=1e-6,atol=1e-7))
            del bf16,fp32,source,out,ref
            torch.cuda.empty_cache()
        report=dict(rank=rank,complete=True,rows=records,lossless_negative_check_passed=True,repeated_changed_data_check_passed=True)
        reports=[None]*4;dist.all_gather_object(reports,report)
        if rank==0:(a.output/'result.json').write_text(json.dumps(dict(complete=True,module_only=True,full_training_qualified=False,ranks=reports,scope=__doc__),indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
