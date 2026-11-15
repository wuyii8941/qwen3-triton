"""Real four-GPU peer-buffer release/recreate and sticky-error tests."""
from datetime import timedelta
import argparse,json,os
from pathlib import Path
import torch
import torch.distributed as dist
from fsdp_peer_rs_v13 import FSDPPeerReduceScatter,CAPACITY


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
    torch.manual_seed(20261115+rank);torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    dist.init_process_group('nccl',timeout=timedelta(minutes=5),device_id=torch.device('cuda',torch.cuda.current_device()))
    try:
        comm=FSDPPeerReduceScatter();flag_ptr=comm.error.data_ptr()
        source=torch.randn(CAPACITY,device='cuda',dtype=torch.bfloat16).float()
        out=torch.empty(CAPACITY//4,device='cuda',dtype=torch.float32);reference=torch.empty_like(out)
        checks=[]
        for cycle in range(2):
            comm(out,source,dist.group.WORLD,dist.ReduceOp.AVG)
            dist.reduce_scatter_tensor(reference,source,op=dist.ReduceOp.AVG)
            assert bool(torch.allclose(out,reference,rtol=1e-6,atol=1e-7)) and torch.isfinite(out).all()
            with comm.validation_memory():
                assert comm.peer is None and comm.error.data_ptr()==flag_ptr
                scratch=torch.empty(1024,device='cuda',dtype=torch.float32);scratch.fill_(cycle);del scratch
            assert comm.peer is not None and comm.peer.error.data_ptr()==flag_ptr
            checks.append(dict(cycle=cycle,peer_release_verified=True,peer_recreated=True,error_storage_unchanged=True))
            source.neg_()
        comm(out,source,dist.group.WORLD,dist.ReduceOp.AVG)
        dist.reduce_scatter_tensor(reference,source,op=dist.ReduceOp.AVG)
        assert bool(torch.allclose(out,reference,rtol=1e-6,atol=1e-7))
        positive=comm.report();assert positive['calls']==3 and positive['checked_elements']==3*CAPACITY and positive['lossless']
        # Deliberately set the shared flag while storage is suspended. Restore
        # must refuse it, rather than silently creating a fresh zero counter.
        rejected=False
        try:
            with comm.validation_memory():comm.error.fill_(1)
        except RuntimeError as exc:
            assert 'Lossless violation persists' in str(exc);rejected=True
        assert rejected and int(comm.error)==1 and comm.peer is None
        row=dict(rank=rank,complete=True,cycles=checks,positive_report=positive,sticky_error_negative_case_rejected=True)
        rows=[None]*4;dist.all_gather_object(rows,row)
        if rank==0:
            with a.output.open('x') as f:f.write(json.dumps(dict(complete=True,full_training_qualified=False,ranks=rows,scope=__doc__),indent=2)+'\n')
            print(json.dumps(dict(complete=True,release_recreate_all_ranks=True,sticky_error_negative_check=True)),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
