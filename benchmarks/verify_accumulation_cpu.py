"""Two-rank CPU DDP objective/sync proof, NOT a CUDA FSDP performance test."""
import argparse
import copy
import json
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from training.fsdp_accumulation import (gradient_sync,sync_this_microbatch,
    shifted_valid_tokens,global_weighted_causal_loss)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);dist.init_process_group('gloo',timeout=timedelta(minutes=2))
    try:
        rank,world=dist.get_rank(),dist.get_world_size();error=[None]
        if rank==0:
            try:a.output.mkdir(parents=True,exist_ok=False)
            except Exception as exc:error[0]=str(exc)
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise RuntimeError(error[0])
        torch.manual_seed(23);initial=torch.nn.Linear(5,7)
        generator=torch.Generator().manual_seed(31+rank)
        x=torch.randn(2,6,5,generator=generator);labels=torch.randint(7,(2,6),generator=generator)
        if rank==0:labels[0]=-100;labels[1,3:]=-100
        else:labels[0,4:]=-100;labels[1,2:]=-100
        count=shifted_valid_tokens(labels);dist.all_reduce(count);count=int(count)
        outputs={}
        for mode in ('large_batch','each','defer'):
            model=DDP(copy.deepcopy(initial));optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            optimizer.zero_grad(set_to_none=True);loss_total=torch.zeros(())
            parts=1 if mode=='large_batch' else 2
            for i in range(parts):
                xx=x if parts==1 else x[i:i+1];yy=labels if parts==1 else labels[i:i+1]
                with gradient_sync(model,True if parts==1 else sync_this_microbatch(i,parts,mode)):
                    loss=global_weighted_causal_loss(model(xx),yy,global_tokens=count,world=world)
                    loss.backward();loss_total+=loss.detach()
            dist.all_reduce(loss_total);loss_total/=world
            gradients=[p.grad.clone() for p in model.parameters()];optimizer.step()
            outputs[mode]=(loss_total,gradients,[p.detach().clone() for p in model.parameters()])
            del model,optimizer
        for mode in ('each','defer'):
            torch.testing.assert_close(outputs[mode][0],outputs['large_batch'][0],atol=1e-7,rtol=1e-6)
            for actual,expected in zip(outputs[mode][1]+outputs[mode][2],outputs['large_batch'][1]+outputs['large_batch'][2]):
                torch.testing.assert_close(actual,expected,atol=2e-7,rtol=1e-5)
        if rank==0:
            result=dict(passed=True,world=world,global_tokens=count,modes=list(outputs),
                cases='Unequal masks and all-ignored local microbatch; gradients and AdamW update',
                scope='CPU DDP objective/sync gate only; CUDA FSDP and performance unverified')
            (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
