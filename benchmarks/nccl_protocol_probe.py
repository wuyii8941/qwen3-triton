"""Isolated four-rank NCCL protocol diagnostic, never a training-speed claim."""
import argparse
import contextlib
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist


def layer_parameter_count(hidden=4096,intermediate=12288,q_heads=32,kv_heads=8,head_dim=128):
    # Bias-free Q/K/V/O, SwiGLU, two hidden norms and shared-head Q/K norms.
    return hidden*head_dim*(2*q_heads+2*kv_heads)+3*hidden*intermediate+2*hidden+2*head_dim


def make_buffers(numel,world,rank,operation,device):
    if world<2 or not 0<=rank<world or numel%world or numel//world<3:
        raise ValueError('Evenly divisible nontrivial four-rank payload required')
    local=numel//world
    if operation=='all_gather':
        inp=torch.full((local,),rank+1,device=device,dtype=torch.bfloat16)
        inp[0]=rank+8;inp[-1]=-(rank+1)
        out=torch.empty(numel,device=device,dtype=inp.dtype)
    elif operation=='reduce_scatter':
        inp=torch.full((numel,),rank+1,device=device,dtype=torch.float32)
        for chunk in range(world):
            inp[chunk*local]=(rank+1)*(chunk+1);inp[(chunk+1)*local-1]=-(rank+1)*(chunk+1)
        out=torch.empty(local,device=device,dtype=inp.dtype)
    else:raise ValueError('Unknown collective')
    return inp,out


def check_output(out,world,rank,operation):
    if operation=='all_gather':
        return all(part[0].item()==r+8 and part[-1].item()==-(r+1)
            and (part[1:-1]==r+1).all().item() for r,part in enumerate(out.chunk(world)))
    if operation=='reduce_scatter':
        total=world*(world+1)//2
        return bool(out[0].item()==total*(rank+1) and out[-1].item()==-total*(rank+1)
                    and (out[1:-1]==total).all().item())
    raise ValueError('Unknown collective')


def profile_collective(collective,synchronize,profiler,cpu_barrier):
    # Profiling is outside timing. Other ranks must not start the next NCCL
    # kernel while rank0 starts/stops CUPTI: such kernels can wait for rank0
    # while profiler finalization waits for device activity to become idle.
    # A NCCL barrier here would itself enqueue waiting GPU work, so use Gloo.
    with profiler:
        cpu_barrier()
        collective();synchronize()
        cpu_barrier()
    cpu_barrier()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-proto',choices=['auto','Simple'],required=True)
    p.add_argument('--steps',type=int,default=20);p.add_argument('--warmup',type=int,default=5);a=p.parse_args()
    if min(a.steps,a.warmup)<1:p.error('Positive warmup and measured steps required')
    proto=os.environ.get('NCCL_PROTO')
    initial_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (a.expected_proto=='auto' and proto not in (None,'')) or (a.expected_proto=='Simple' and proto!='Simple'):
        raise ValueError('Actual NCCL_PROTO does not match experiment label')
    torch.set_num_threads(1);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=5),
        device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    rank,world=dist.get_rank(),dist.get_world_size();error=[None]
    cpu_group=dist.new_group(backend='gloo',timeout=timedelta(minutes=5))
    if rank==0:
        try:a.output.mkdir(parents=True,exist_ok=False)
        except Exception as exc:error[0]=str(exc)
    dist.broadcast_object_list(error,src=0,group=cpu_group)
    if error[0]:raise RuntimeError(error[0])
    rows=[]
    try:
        if world!=4:raise ValueError('Four-rank hardware diagnostic required')
        def cpu_barrier():dist.barrier(group=cpu_group)
        sizes=dict(small=1<<20,decoder=layer_parameter_count(),head=151936*4096)
        for name,numel in sizes.items():
            for operation in ('all_gather','reduce_scatter'):
                inp,out=make_buffers(numel,world,rank,operation,'cuda')
                # Finish allocations/event initialization on EVERY rank before
                # another rank can launch a spinning Simple-protocol kernel.
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();end.record();torch.cuda.synchronize();cpu_barrier()
                def collective():
                    if operation=='all_gather':dist.all_gather_into_tensor(out,inp)
                    else:dist.reduce_scatter_tensor(out,inp,op=dist.ReduceOp.SUM)
                def verify():
                    ok=torch.tensor(int(check_output(out,world,rank,operation)),device='cpu')
                    dist.all_reduce(ok,op=dist.ReduceOp.MIN,group=cpu_group)
                    if not ok.item():raise AssertionError('Collective payload/sentinel mismatch')
                collective();torch.cuda.synchronize();verify()
                for _ in range(a.warmup):collective()
                torch.cuda.synchronize()
                cpu_barrier()
                timings=[]
                for _ in range(a.steps):
                    cpu_barrier();torch.cuda.synchronize()
                    tick=time.perf_counter();start.record();collective();end.record();torch.cuda.synchronize()
                    wall=(time.perf_counter()-tick)*1000;gpu=start.elapsed_time(end)
                    values=torch.tensor([gpu,wall],device='cpu',dtype=torch.float64)
                    dist.all_reduce(values,op=dist.ReduceOp.MAX,group=cpu_group)
                    timings.append(dict(max_rank_cuda_ms=values[0].item(),max_rank_wall_ms=values[1].item()))
                verify();cpu_barrier();torch.cuda.synchronize()
                profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) if rank==0 else contextlib.nullcontext()
                profile_collective(collective,torch.cuda.synchronize,profiler,
                    cpu_barrier)
                if rank==0:
                    path=a.output/f'{name}_{operation}_trace.json';profiler.export_chrome_trace(str(path))
                    trace=path.read_bytes();events=json.loads(trace)['traceEvents']
                    kernels=sorted({e['name'] for e in events if e.get('cat')=='kernel' and
                        'nccl' in e.get('name','').lower() and operation.replace('_','') in e.get('name','').lower()})
                    if not kernels:raise ValueError('Missing actual collective kernel evidence')
                    if a.expected_proto=='Simple' and not all('simple' in n.lower() for n in kernels):
                        raise ValueError('Requested Simple protocol not verified by kernel trace')
                    ms=statistics.median(t['max_rank_cuda_ms'] for t in timings)
                    payload=numel*(2 if operation=='all_gather' else 4)
                    rows.append(dict(size=name,global_parameter_numel=numel,operation=operation,
                        dtype=str(inp.dtype),payload_bytes=payload,timings=timings,median_cuda_ms=ms,
                        algorithm_GB_s=payload/(ms/1000)/1e9,
                        estimated_ring_bus_GB_s=payload*(world-1)/world/(ms/1000)/1e9,
                        correctness_before_and_after=True,observed_kernels=kernels,
                        trace_source=str(path),trace_sha256=hashlib.sha256(trace).hexdigest()))
                # Trace export/parsing is rank0-only. No rank may allocate the
                # next payload or enqueue another NCCL call until it completes.
                cpu_barrier()
                del inp,out,start,end
        unchanged=torch.tensor(int(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==initial_source),device='cpu')
        dist.all_reduce(unchanged,op=dist.ReduceOp.MIN,group=cpu_group)
        if not unchanged.item():raise ValueError('Probe implementation changed during timing')
        if rank==0:
            report=dict(complete=True,world_size=world,expected_proto=a.expected_proto,
                warmup=a.warmup,steps=a.steps,records=rows,pytorch=torch.__version__,cuda=torch.version.cuda,
                nccl=torch.cuda.nccl.version(),gpu=torch.cuda.get_device_name(),
                environment={k:os.environ.get(k) for k in ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE',
                    'NCCL_SHM_DISABLE','NCCL_P2P_LEVEL','NCCL_CONF_FILE')},
                source_sha256=initial_source,
                profile_synchronization='CPU Gloo synchronization for allocations, metrics, profiler and trace export; only tested collective uses NCCL',
                scope='Isolated collective correctness/latency/bandwidth plus extra untimed protocol traces; not model speed or convergence')
            (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(complete=True,protocol=a.expected_proto,records=len(rows))),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
