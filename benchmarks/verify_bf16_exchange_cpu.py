"""Gloo transport/reduction oracle only; no GPU or training-speed claim."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from benchmarks.bf16_exchange_reduce import exchange_reduce


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);dist.init_process_group('gloo')
    rank,world=dist.get_rank(),dist.get_world_size()
    try:
        if world not in (2,4):raise ValueError('Two or four CPU ranks required')
        error=[None]
        if rank==0:
            try:a.output.mkdir(parents=True,exist_ok=False)
            except Exception as exc:error[0]=str(exc)
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise RuntimeError(error[0])
        torch.manual_seed(900+rank)
        rows=[]
        for count in (world*129,4096):
            gradient=torch.randn(count,dtype=torch.bfloat16)
            gradient[0]=-0.;gradient[1]=rank+1
            sources=[torch.empty_like(gradient) for _ in range(world)]
            dist.all_gather(sources,gradient)
            for average in (False,True):
                result,received=exchange_reduce(gradient,average=average,return_received=True)
                local=count//world
                for source in range(world):
                    expected=sources[source][rank*local:(rank+1)*local]
                    assert torch.equal(received.reshape(world,local)[source].view(torch.int16),expected.view(torch.int16))
                values=torch.stack([g[rank*local:(rank+1)*local] for g in sources]).float()
                expected=(values/world if average else values).sum(dim=0)
                torch.testing.assert_close(result,expected,atol=0,rtol=0)
                rows.append(dict(count=count,average=average,transport_bitwise_equal=True,fp32_oracle_exact=True))
        reports=[None]*world
        dist.all_gather_object(reports,dict(rank=rank,records=rows))
        if rank==0:
            report=dict(passed=True,world_size=world,backend='gloo',records=reports,
                scope='BF16 transport and FP32 oracle only; Triton/GPU/NCCL gradient parity, memory and performance unproven')
            (a.output/'result.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(passed=True,scope=report['scope'])),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
