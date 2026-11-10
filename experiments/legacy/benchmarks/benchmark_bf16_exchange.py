"""Four-rank comm replacement test from an EXACT BF16-upcast FP32 buffer.

Packing back to BF16 is included in exchange timing. Workspace is prepared
outside timing, so this is not a complete FSDP/allocator/optimizer benchmark.
No arbitrary FP32 gradient quantization and no automatic model integration.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

import torch
import torch.distributed as dist
import triton

from benchmarks.bf16_exchange_reduce import _sum_contributions
from benchmarks.nccl_protocol_probe import layer_parameter_count


MODES=('native_fp32','exchange_torch','exchange_triton')
SIZES=dict(small=(1<<20)+4,decoder=layer_parameter_count(),head=151936*4096)
INPUT_CONTRACT='FP32 exact upcast from original BF16; no accumulation; random finite gradients'


def source_hashes():
    files=[Path(__file__).resolve(),Path(__file__).with_name('bf16_exchange_reduce.py').resolve(),
           Path(__file__).with_name('nccl_protocol_probe.py').resolve()]
    return {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def audit_report(report,expected_proto):
    if (not report.get('complete') or report.get('world_size')!=4
            or report.get('expected_proto')!=expected_proto or report.get('steps',0)<12
            or report.get('warmup',0)<3 or report.get('source_hashes')!=source_hashes()):
        raise ValueError('Complete matching four-rank experiment with current source required')
    if report.get('input_contract')!=INPUT_CONTRACT or report.get('protocol_verified_by_trace') is not False:
        raise ValueError('Input/protocol evidence contract mismatch')
    actual=report['environment'].get('NCCL_PROTO')
    if (expected_proto=='auto' and actual not in (None,'')) or (expected_proto=='Simple' and actual!='Simple'):
        raise ValueError('Protocol request mismatch')
    sizes=report['selected_sizes']
    if not sizes or len(set(sizes))!=len(sizes) or any(s not in SIZES for s in sizes):
        raise ValueError('Explicit distinct supported sizes required')
    for size in sizes:
        fingerprints=report['input_prefix_samples'][size]
        if len(fingerprints)!=4 or any(not re.fullmatch('[0-9a-f]{64}',s) for s in fingerprints):
            raise ValueError('Four input-prefix hashes required')
    expected_order=[(s,r,m) for s in sizes for r,order in ((0,MODES),(1,tuple(reversed(MODES)))) for m in order]
    if [(r['size'],r['repeat'],r['mode']) for r in report['records']]!=expected_order:
        raise ValueError('Reverse-order case sequence mismatch')
    for row in report['records']:
        n=SIZES[row['size']];mode=row['mode'];samples=row['timings']
        if (row['numel']!=n or row.get('correctness_before_and_after') is not True
                or len(samples)!=report['steps']):raise ValueError('Shape/correctness/sample count mismatch')
        native=mode=='native_fp32'
        if (row['wire_element_dtype']!=('torch.float32' if native else 'torch.bfloat16')
                or row['wire_representation_bytes']!=n*(4 if native else 2)
                or row['network_bytes_measured'] is not False):raise ValueError('Wire representation mismatch')
        for key,reported in (('max_rank_cuda_ms','median_cuda_ms'),('max_rank_wall_ms','median_wall_ms')):
            values=[s[key] for s in samples]
            if any(not math.isfinite(x) or x<=0 for x in values):raise ValueError('Invalid raw timing')
            if not math.isclose(statistics.median(values),row[reported],rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('Median does not match raw timing')
        expected_workspace=n if mode=='native_fp32' else n*(9 if mode=='exchange_torch' else 5)
        if row['workspace_tensor_bytes']!=expected_workspace or row['peak_extra_allocated_bytes']<0:
            raise ValueError('Workspace accounting mismatch')
    calculated=comparisons(report['records'],sizes)
    if any(report.get(key)!=value for key,value in calculated.items()):raise ValueError('Decision differs from raw measurements')
    return calculated


def comparisons(records,sizes):
    expected={(size,r,mode) for size in sizes for r in (0,1) for mode in MODES}
    indexed={(r['size'],r['repeat'],r['mode']):r['median_cuda_ms'] for r in records}
    if len(records)!=len(expected) or set(indexed)!=expected:
        raise ValueError('Complete distinct reverse-order records required')
    if any(not math.isfinite(t) or t<=0 for t in indexed.values()):raise ValueError('Invalid latency')
    ratios={size:{mode:[indexed[size,r,'native_fp32']/indexed[size,r,mode] for r in (0,1)]
        for mode in MODES[1:]} for size in sizes}
    return dict(paired_throughput_ratios=ratios,triton_over_two_percent_in_both_orders={
        size:all(t>1.02 for t in ratios[size]['exchange_triton']) for size in sizes})


def make_mode(mode,source,world):
    cols=source.numel()//world
    output=torch.empty(cols,device=source.device,dtype=torch.float32)
    if mode=='native_fp32':
        buffers=[output]
        def run():
            dist.reduce_scatter_tensor(output,source,op=dist.ReduceOp.AVG)
            return output
    else:
        packed=torch.empty(source.numel(),device=source.device,dtype=torch.bfloat16)
        received=torch.empty_like(packed);buffers=[packed,received,output]
        scratch=torch.empty_like(source) if mode=='exchange_torch' else None
        if scratch is not None:buffers.append(scratch)
        if mode not in MODES:raise ValueError('Unknown communication mode')
        def run():
            packed.copy_(source)  # timed cast of exact BF16-upcast values
            dist.all_to_all_single(received,packed)
            if mode=='exchange_torch':
                scratch.copy_(received);scratch.mul_(1/world)
                torch.sum(scratch.reshape(world,cols),dim=0,out=output)
            else:
                _sum_contributions[(triton.cdiv(cols,1024),)](received,output,cols,world,
                    1/world,triton.next_power_of_2(world),1024,num_warps=4)
            return output
    return run,buffers


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-proto',choices=['auto','Simple'],required=True)
    p.add_argument('--sizes',nargs='+',choices=list(SIZES),default=list(SIZES))
    p.add_argument('--steps',type=int,default=20);p.add_argument('--warmup',type=int,default=5)
    a=p.parse_args()
    if min(a.steps,a.warmup)<1 or len(set(a.sizes))!=len(a.sizes):p.error('Positive counts and distinct sizes required')
    proto=os.environ.get('NCCL_PROTO')
    if (a.expected_proto=='auto' and proto not in (None,'')) or (a.expected_proto=='Simple' and proto!='Simple'):
        raise ValueError('Actual protocol environment differs from declared request')
    sources=source_hashes()
    torch.set_num_threads(1);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=3),
        device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    rank,world=dist.get_rank(),dist.get_world_size()
    cpu=dist.new_group(backend='gloo',timeout=timedelta(minutes=3))
    def barrier():dist.barrier(group=cpu)
    def max_cpu(values):
        tensor=torch.tensor(values,dtype=torch.float64)
        dist.all_reduce(tensor,op=dist.ReduceOp.MAX,group=cpu)
        return tensor.tolist()
    report=dict(complete=False,records=[],expected_proto=a.expected_proto,world_size=world,
        pytorch=str(torch.__version__),cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
        gpu=torch.cuda.get_device_name(),source_hashes=sources,steps=a.steps,warmup=a.warmup,
        input_seed_base=20261108,input_prefix_samples={},selected_sizes=a.sizes,
        memory_measurement='Steady allocated peak minus warmed baseline excluding known private workspace; not reserved/NVML memory',
        input_contract=INPUT_CONTRACT,
        protocol_verified_by_trace=False,
        environment={k:os.environ.get(k) for k in ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE')},
        scope='Preallocated operator workspace, cast included, reverse-order microbench; not FSDP step, convergence, full-range bitwise equivalence or protocol trace proof')
    def write():
        if rank==0:(a.output/'result.json').write_text(json.dumps(report,indent=2))
    try:
        if world!=4:raise ValueError('Four CUDA ranks required')
        error=[None]
        if rank==0:
            try:a.output.mkdir(parents=True,exist_ok=False)
            except Exception as exc:error[0]=str(exc)
        dist.broadcast_object_list(error,src=0,group=cpu)
        if error[0]:raise RuntimeError(error[0])
        for size in a.sizes:
            count=SIZES[size]
            torch.cuda.manual_seed(20261108+rank)
            original=torch.randn(count,device='cuda',dtype=torch.bfloat16)
            source=original.float();del original
            sample=hashlib.sha256(source[:min(count,4096)].cpu().numpy().tobytes()).hexdigest()
            samples=[None]*world;dist.all_gather_object(samples,sample,group=cpu)
            if rank==0:report['input_prefix_samples'][size]=samples
            reference=torch.empty(count//world,device='cuda',dtype=torch.float32)
            torch.cuda.synchronize();barrier()
            dist.reduce_scatter_tensor(reference,source,op=dist.ReduceOp.AVG)
            torch.cuda.synchronize();barrier()
            for repeat,order in ((0,MODES),(1,tuple(reversed(MODES)))):
                for mode in order:
                    barrier();torch.cuda.synchronize()
                    run,buffers=make_mode(mode,source,world)
                    workspace_bytes=sum(b.numel()*b.element_size() for b in buffers)
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();end.record();torch.cuda.synchronize();barrier()
                    tick=time.perf_counter();run();torch.cuda.synchronize()
                    startup=max_cpu([time.perf_counter()-tick])[0]
                    torch.testing.assert_close(run(),reference,atol=1e-6,rtol=1e-6)
                    for _ in range(a.warmup):run()
                    torch.cuda.synchronize();barrier()
                    # Earlier cases' NCCL references can retire during warmup.
                    # Use the WARMED baseline, retaining this case's explicitly
                    # allocated private workspace in the reported peak delta.
                    base=torch.cuda.memory_allocated()-workspace_bytes
                    torch.cuda.reset_peak_memory_stats()
                    timings=[]
                    for _ in range(a.steps):
                        barrier();torch.cuda.synchronize()
                        tick=time.perf_counter();start.record();run();end.record();torch.cuda.synchronize()
                        values=max_cpu([start.elapsed_time(end),(time.perf_counter()-tick)*1000])
                        timings.append(dict(max_rank_cuda_ms=values[0],max_rank_wall_ms=values[1]))
                    peak=max_cpu([torch.cuda.max_memory_allocated()-base])[0]
                    torch.testing.assert_close(run(),reference,atol=1e-6,rtol=1e-6)
                    torch.cuda.synchronize();barrier()
                    row=dict(size=size,numel=count,repeat=repeat,mode=mode,timings=timings,
                        wire_element_dtype='torch.float32' if mode=='native_fp32' else 'torch.bfloat16',
                        wire_representation_bytes=count*(4 if mode=='native_fp32' else 2),
                        network_bytes_measured=False,
                        median_cuda_ms=statistics.median(r['max_rank_cuda_ms'] for r in timings),
                        median_wall_ms=statistics.median(r['max_rank_wall_ms'] for r in timings),
                        peak_extra_allocated_bytes=int(peak),
                        workspace_tensor_bytes=workspace_bytes,
                        first_call_max_rank_seconds=startup,correctness_before_and_after=True)
                    if rank==0:
                        report['records'].append(row);write()
                        print(json.dumps({k:v for k,v in row.items() if k!='timings'}),flush=True)
                    barrier();del run,buffers,start,end
            if hashlib.sha256(source[:min(count,4096)].cpu().numpy().tobytes()).hexdigest()!=sample:
                raise ValueError('Common input prefix mutated during benchmark')
            del source,reference;torch.cuda.synchronize();barrier()
        unchanged=torch.tensor(int(source_hashes()==sources),dtype=torch.int32)
        dist.all_reduce(unchanged,op=dist.ReduceOp.MIN,group=cpu)
        if not unchanged.item():raise ValueError('Communication experiment source changed')
        if rank==0:
            report.update(complete=True,**comparisons(report['records'],a.sizes));write()
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}');write();raise
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
