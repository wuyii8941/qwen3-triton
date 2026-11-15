"""Four-GPU full-coordinate FP32 RS test at real decoder message sizes.

Timing includes layout packing and scratch allocation, not optimizer/model
execution. All modes share deterministic fill=True. This is not a training
speedup claim. Native/separate/hierarchical run in both orders.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import torch
import torch.distributed as dist
from training.fsdp.communication_v13 import ExperimentalReduceScatter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2)
    torch.manual_seed(20261115+int(os.environ['RANK']))
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    dist.init_process_group('nccl', timeout=timedelta(minutes=10), device_id=torch.device('cuda',torch.cuda.current_device()))
    try:
        rank=dist.get_rank(); assert dist.get_world_size()==4
        if rank==0:args.output.mkdir(exist_ok=False)
        dist.barrier()
        separate=ExperimentalReduceScatter('separate')
        hierarchical=ExperimentalReduceScatter('hierarchical')
        rows=[]
        # Actual Qwen3-8B decoder layer parameters: Q/K/V/O, gate/up/down,
        # two hidden norms and two head-dimension Q/K norms.
        layer_parameters=2*4096*4096+2*1024*4096+3*12288*4096+2*4096+2*128
        for n in [2**20,16*2**20,layer_parameters]:
            assert n%4==0
            x=torch.randn(n,device='cuda',dtype=torch.float32)
            ref=torch.empty(n//4,device='cuda',dtype=torch.float32)
            got=torch.empty_like(ref)
            errors={}
            for op_name,op in [('AVG',dist.ReduceOp.AVG),('SUM',dist.ReduceOp.SUM)]:
                dist.reduce_scatter_tensor(ref,x,op=op)
                for name,comm in [('separate',separate),('hierarchical',hierarchical)]:
                    comm(got,x,dist.group.WORLD,op)
                    delta=got-ref
                    relative=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref).clamp_min(1e-12))
                    maximum=float(delta.abs().max())
                    passed=relative<1e-6 and maximum<2e-6 and bool(torch.isfinite(got).all())
                    assert passed,(n,op_name,name,relative,maximum)
                    errors[name+'_'+op_name]=dict(relative_l2=relative,max_abs=maximum,passed=passed,all_coordinates=True)
                    del delta
            del ref
            for order,modes in enumerate([['native','separate','hierarchical'],['hierarchical','separate','native']],1):
                for mode in modes:
                    comm=dict(separate=separate,hierarchical=hierarchical).get(mode)
                    def run():
                        if comm is None:dist.reduce_scatter_tensor(got,x,op=dist.ReduceOp.AVG)
                        else:comm(got,x,dist.group.WORLD,dist.ReduceOp.AVG)
                    for _ in range(3):run()
                    samples=[]
                    for _ in range(8):
                        torch.cuda.synchronize();dist.barrier();start=time.perf_counter()
                        run();torch.cuda.synchronize()
                        elapsed=torch.tensor(time.perf_counter()-start,device='cuda',dtype=torch.float64)
                        dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
                        samples.append(float(elapsed))
                    row=dict(input_numel=n,input_bytes=n*4,order=order,mode=mode,samples_seconds=samples,
                             median_seconds=statistics.median(samples),errors=errors)
                    rows.append(row)
                    if rank==0:print(json.dumps(row),flush=True)
            del x,got
            torch.cuda.empty_cache()
        report=dict(rank=rank,complete=True,rows=rows,communication=[separate.report(),hierarchical.report()])
        (args.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        reports=[None]*4;dist.all_gather_object(reports,report)
        if rank==0:
            files=[Path(__file__),Path(__file__).resolve().parents[1]/'training/fsdp/communication_v13.py']
            result=dict(complete=True,full_training_qualified=False,torch=torch.__version__,NCCL=torch.cuda.nccl.version(),
                ranks=reports,topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
                sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},scope=__doc__)
            (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
