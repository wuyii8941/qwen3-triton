"""Diagnose native FP32 sum order at actual vocabulary CE/dX sizes.

All coordinate comparisons, not an accuracy/adoption gate. Finds matching CPU
addition trees without changing the fixed full-model gradient tolerances.
"""
import argparse,itertools,json,os
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from qwen3_triton.runtime.collectives import TritonCollectives


def trees(stack,expected):
    rows=[]
    for p in itertools.permutations(range(4)):
        got=((stack[p[0]]+stack[p[1]])+stack[p[2]])+stack[p[3]]
        rows.append(dict(tree=['left',*p],different_bits=int((got.view(torch.int32)!=expected.view(torch.int32)).sum())))
    for p in [(0,1,2,3),(0,2,1,3),(0,3,1,2)]:
        got=(stack[p[0]]+stack[p[1]])+(stack[p[2]]+stack[p[3]])
        rows.append(dict(tree=['balanced',*p],different_bits=int((got.view(torch.int32)!=expected.view(torch.int32)).sum())))
    return sorted(rows,key=lambda r:r['different_bits'])


def addition_segments(stack,expected):
    candidates=[]
    for p in itertools.permutations(range(4)):
        candidates.append((['left',*p],((stack[p[0]]+stack[p[1]])+stack[p[2]])+stack[p[3]]))
    for p in [(0,1,2,3),(0,2,1,3),(0,3,1,2)]:
        candidates.append((['balanced',*p],(stack[p[0]]+stack[p[1]])+(stack[p[2]]+stack[p[3]])))
    masks=torch.stack([x.view(torch.int32)!=expected.view(torch.int32) for _,x in candidates])
    segments=[]
    for begin in range(0,expected.numel(),32):
        end=min(begin+32,expected.numel());errors=masks[:,begin:end].sum(1)
        matched=[candidates[i][0] for i in range(len(candidates)) if errors[i]==0]
        if segments and segments[-1]['matching_trees']==matched and segments[-1]['best_different_bits']==int(errors.min()):segments[-1]['end']=end
        else:segments.append(dict(begin=begin,end=end,matching_trees=matched,best_different_bits=int(errors.min())))
    return segments


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=timedelta(minutes=10))
    cpu=dist.new_group(backend='gloo',timeout=timedelta(minutes=10))
    try:
        comm=TritonCollectives(group=cpu,capacities=dict(gather=4096,scatter=40*2**20,reduce=1*2**20,alltoall=4096))
        results=[]
        for kind,n in [('all_reduce',4088),('all_reduce',16352),('reduce_scatter',4*512*4096)]:
            hosts=[]
            for r in range(4):
                gen=torch.Generator().manual_seed(17000+r)
                # Positive exponentials plus signed dX-like values distinguish
                # rounding trees; no GPU reference computation except NCCL.
                value=torch.rand(n,generator=gen,dtype=torch.float32)
                if kind=='reduce_scatter':value=(value-.5)*(.1 if r%2 else 10.)
                hosts.append(value)
            source=hosts[rank].cuda();candidate=torch.empty(n if kind=='all_reduce' else n//4,device='cuda')
            if kind=='all_reduce':
                native=source.clone()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if rank==0 else nullcontext() as trace:
                    dist.all_reduce(native);torch.cuda.synchronize()
                native_trace=None
                if rank==0:
                    path=a.output/f'native_all_reduce_{n}.json.gz';trace.export_chrome_trace(str(path))
                    import gzip
                    native_trace=dict(Counter(e['name'] for e in json.loads(gzip.decompress(path.read_bytes()))['traceEvents'] if e.get('cat')=='kernel'))
                candidate.copy_(source);comm.all_reduce(candidate)
                stack=torch.stack(hosts)
            else:
                native=torch.empty(n//4,device='cuda');dist.reduce_scatter_tensor(native,source,op=dist.ReduceOp.SUM)
                comm.reduce_scatter(candidate,source,'sum');stack=torch.stack([x[rank*(n//4):(rank+1)*(n//4)] for x in hosts])
            got=candidate.cpu();expected=native.cpu();orders=trees(stack,expected)
            results.append(dict(kind=kind,input_elements=n,output_elements=expected.numel(),
                candidate_different_bits=int((got.view(torch.int32)!=expected.view(torch.int32)).sum()),
                BF16_cast_different_bits=int((got.bfloat16().view(torch.int16)!=expected.bfloat16().view(torch.int16)).sum()),
                candidate_max_abs=float((got-expected).abs().max()),CPU_tree_matches=orders,
                addition_segments=addition_segments(stack,expected) if kind=='all_reduce' else None,
                native_trace=native_trace if kind=='all_reduce' else None))
            del hosts,source,candidate,native,stack,got,expected
        transport=comm.report();comm.close()
        row=dict(rank=rank,complete=True,results=results,communication=transport)
        rows=[None]*4;dist.all_gather_object(rows,row,group=cpu)
        if rank==0:(a.output/'check.json').write_text(json.dumps(dict(complete=True,kind='reduction_order',full_model_qualified=False,ranks=rows,scope=__doc__),indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
