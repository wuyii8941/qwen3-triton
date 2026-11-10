"""CPU/Gloo sharded-payload gate, not real-model resume or I/O speed proof."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor,Shard,Replicate

from benchmarks.dcp_sorted_reader import OffsetSortedFileSystemReader


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);dist.init_process_group('gloo')
    rank,world=dist.get_rank(),dist.get_world_size()
    try:
        if world!=2:raise ValueError('Two CPU ranks required for the uneven-shard gate')
        error=[None]
        if rank==0:
            try:a.output.mkdir(parents=True,exist_ok=False)
            except Exception as exc:error[0]=str(exc)
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise RuntimeError(error[0])
        mesh=init_device_mesh('cpu',(world,))
        def payload(zero=False):
            weight=torch.arange(35,dtype=torch.float32).reshape(5,7)
            source=dict(weight=weight,exp_avg=weight*.125,exp_avg_sq=weight*.25,
                        norm=torch.arange(7,dtype=torch.float32),step=torch.tensor(2.))
            return {**{k:distribute_tensor(torch.zeros_like(v) if zero else v,mesh,
                    [Replicate() if v.ndim==0 else Shard(0)]) for k,v in source.items()},
                    'metadata':dict(tag='' if zero else 'sharded-payload',step=0 if zero else 2)}
        source=payload()
        dcp.save(source,checkpoint_id=a.output/'state')
        destination=payload(zero=True)
        reader=OffsetSortedFileSystemReader(a.output/'state')
        dcp.load(destination,storage_reader=reader)
        checks={}
        for key,value in source.items():
            actual=destination[key]
            if isinstance(value,torch.Tensor):
                assert value.placements==actual.placements and value.shape==actual.shape and value.dtype==actual.dtype
                torch.testing.assert_close(value.to_local(),actual.to_local(),rtol=0,atol=0)
                checks[key]=dict(shape=list(actual.shape),local_shape=list(actual.to_local().shape),
                    sha256=hashlib.sha256(actual.to_local().contiguous().numpy().tobytes()).hexdigest())
            elif value!=actual:raise AssertionError('Byte metadata mismatch')
        reports=[None]*world
        dist.all_gather_object(reports,dict(rank=rank,passed=True,checks=checks))
        if rank==0:
            report=dict(passed=True,world_size=world,backend='gloo',records=reports,
                pytorch=str(torch.__version__),
                scope='CPU sharded payload/replicated scalar/byte metadata roundtrip; not optimizer trajectory, GPU/8B resume or performance proof')
            (a.output/'result.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(passed=True,scope=report['scope'])),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
