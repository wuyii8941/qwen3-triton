"""Independent DCP interrupt/resume replay gate; CPU or real CUDA FSDP2.

CPU Gloo test is not evidence of CUDA FSDP correctness. CUDA mode explicitly
uses a D128 tiny Qwen with retained forward parameters. Never loads 8B here.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from training.fsdp_checkpoint import save_checkpoint,load_checkpoint


def local(value):return value.to_local() if hasattr(value,'to_local') else value


def assert_tree(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(local(a),local(b),atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_tree(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tree(x,y)
    else:assert a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1)
    if args.cuda:torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl' if args.cuda else 'gloo',timeout=timedelta(minutes=5))
    rank=dist.get_rank();error=[None]
    if rank==0:
        try:args.output.mkdir(parents=True,exist_ok=False)
        except Exception as exc:error[0]=str(exc)
    dist.broadcast_object_list(error,src=0)
    if error[0]:raise RuntimeError(error[0])
    dist.barrier()
    try:
        def make_model():
            if args.cuda:
                from transformers import Qwen3Config,Qwen3ForCausalLM
                from training.official_fsdp import shard_official_qwen
                config=Qwen3Config(vocab_size=97,hidden_size=512,intermediate_size=768,
                    num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=128,
                    attention_dropout=0.,tie_word_embeddings=False)
                config._attn_implementation='sdpa'
                return shard_official_qwen(Qwen3ForCausalLM(config),keep_full_layers=1).train()
            return torch.nn.Sequential(torch.nn.Linear(7,11),torch.nn.GELU(),
                torch.nn.Dropout(.2),torch.nn.Linear(11,5)).train()
        torch.manual_seed(123);model=make_model()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        torch.manual_seed(300+rank);random.seed(400+rank);np.random.seed(500+rank)
        def update(m,opt,index):
            opt.zero_grad(set_to_none=True)
            scalar=1.+random.random()+float(np.random.random())
            if args.cuda:
                ids=torch.randint(97,(2,17),device='cuda');loss=m(ids,labels=ids,use_cache=False).loss*scalar
            else:
                x=torch.randn(4,7);target=torch.randn(4,5)
                loss=(m(x)-target).square().mean()*scalar
            loss.backward()
            if not args.cuda:
                for param in m.parameters():dist.all_reduce(param.grad);param.grad.div_(dist.get_world_size())
            opt.step();return loss.detach().cpu()
        for i in range(2):update(model,optimizer,i)
        checkpoint=args.output/'step2';metadata=dict(test='resume_replay_v1',cuda=args.cuda)
        progress=dict(next_block=2,epoch=0,rank=rank)
        save_checkpoint(checkpoint,model,optimizer,step=2,data_state=progress,metadata=metadata)
        expected_losses=[update(model,optimizer,i) for i in range(2,4)]
        expected_parameters={n:local(v).detach().cpu().clone() for n,v in model.named_parameters()}
        expected_optimizer=copy.deepcopy(optimizer.state_dict())
        expected_rng=(torch.rand(3),np.random.random(3),random.random())
        expected_cuda_rng=torch.rand(3,device='cuda') if args.cuda else None
        torch.manual_seed(999);fresh=make_model();fresh_optimizer=torch.optim.AdamW(fresh.parameters(),lr=1e-3)
        restored=load_checkpoint(checkpoint,fresh,fresh_optimizer,metadata=metadata)
        assert restored==dict(step=2,data_state=progress)
        losses=[update(fresh,fresh_optimizer,i) for i in range(2,4)]
        for actual,expected in zip(losses,expected_losses):torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        for name,param in fresh.named_parameters():
            torch.testing.assert_close(local(param).detach().cpu(),expected_parameters[name],atol=0,rtol=0)
        assert_tree(fresh_optimizer.state_dict(),expected_optimizer)
        torch.testing.assert_close(torch.rand(3),expected_rng[0],atol=0,rtol=0)
        np.testing.assert_array_equal(np.random.random(3),expected_rng[1]);assert random.random()==expected_rng[2]
        if args.cuda:torch.testing.assert_close(torch.rand(3,device='cuda'),expected_cuda_rng,atol=0,rtol=0)
        checks=[]
        for label,call in [
            ('overwrite',lambda:save_checkpoint(checkpoint,fresh,fresh_optimizer,step=4,data_state=progress,metadata=metadata)),
            ('metadata',lambda:load_checkpoint(checkpoint,fresh,fresh_optimizer,metadata={'wrong':True})),
            ('incomplete',lambda:load_checkpoint(args.output/'missing',fresh,fresh_optimizer,metadata=metadata))]:
            try:call()
            except RuntimeError:checks.append(label)
            else:raise AssertionError(f'Missing safety check: {label}')
        import training.fsdp_checkpoint as module
        disk_usage=module.shutil.disk_usage
        try:
            module.shutil.disk_usage=lambda path:SimpleNamespace(free=0)
            try:save_checkpoint(args.output/'no_space',fresh,fresh_optimizer,step=4,data_state=progress,metadata=metadata)
            except RuntimeError:checks.append('disk_budget')
            else:raise AssertionError('Missing disk guard')
        finally:module.shutil.disk_usage=disk_usage
        assert not (args.output/'no_space').exists()
        if rank==0:
            report=dict(passed=True,cuda_fsdp=args.cuda,world_size=dist.get_world_size(),
                replay_steps=2,parameter_optimizer_rng_replay='bitwise_equal',negative_gates=checks,
                scope='D128 tiny Qwen CUDA FSDP2' if args.cuda else 'CPU replicated model DCP; NOT a CUDA FSDP gate')
            (args.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
