"""Resource-gated real-checkpoint resume replay; not a convergence job.

Hashes one local tensor at a time rather than retaining a second CPU copy of
the full optimizer. Deterministic correctness mode is not a timing baseline.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.distributed as dist


def tensor_digest(tensor):
    local=tensor.to_local() if hasattr(tensor,'to_local') else tensor
    host=local.detach().cpu().contiguous()
    raw=host.reshape(-1).view(torch.uint8).numpy()
    return dict(shape=list(tensor.shape),local_shape=list(local.shape),dtype=str(local.dtype),
        sha256=hashlib.sha256(memoryview(raw)).hexdigest())


def state_digest(model,optimizer):
    parameters={n:tensor_digest(p) for n,p in model.named_parameters()}
    states={n:{k:tensor_digest(v) if isinstance(v,torch.Tensor) else v
               for k,v in optimizer.state[p].items()} for n,p in model.named_parameters()}
    groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]
    return dict(parameters=parameters,buffers={n:tensor_digest(b) for n,b in model.named_buffers()},
        optimizer_states=states,param_groups=groups)


def resource_plan(model,output):
    from training.fsdp_checkpoint import required_bytes
    required=required_bytes(model)
    available=shutil.disk_usage(output).free
    ram=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:')))*1024
    # DCP CPU-offloads master and moments across all four ranks. Hashes need
    # only one extra local tensor per process, not another global state copy.
    ram_required=sum(p.numel()*12 for p in model.parameters())+16*2**30
    return dict(parameters=sum(p.numel() for p in model.parameters()),
        disk_required_bytes=required,disk_available_bytes=available,
        host_ram_required_bytes=ram_required,host_ram_available_bytes=ram,
        resources_sufficient=available>=required and ram>=ram_required)


def rng_samples():
    return dict(torch_cpu=torch.rand(3).tolist(),torch_cuda=torch.rand(3,device='cuda').cpu().tolist(),
        numpy=np.random.random(3).tolist(),python=[random.random() for _ in range(3)])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path('training/fsdp_convergence_protocol_v1.json'))
    p.add_argument('--resource-plan',action='store_true')
    p.add_argument('--keep-full-layers',type=int,default=24)
    p.add_argument('--checkpoint-every',type=int,default=0)
    p.add_argument('--tiny',action='store_true',help='Synthetic D128 pipeline gate; NOT real 8B proof')
    args=p.parse_args();cfg=json.loads(args.protocol.read_text())
    from transformers import AutoConfig,AutoModelForCausalLM,Qwen3Config,Qwen3ForCausalLM
    if args.tiny:cfg['sequence_length']=17
    def tiny_model():
        config=Qwen3Config(vocab_size=97,hidden_size=512,intermediate_size=768,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=128,
            tie_word_embeddings=False,attention_dropout=0.)
        config._attn_implementation='sdpa'
        return Qwen3ForCausalLM(config)
    if args.resource_plan:
        with torch.device('meta'):
            model=tiny_model() if args.tiny else AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(cfg['model'],local_files_only=True))
        # Read-only: caller must choose an existing data-disk output parent.
        print(json.dumps(resource_plan(model,args.output.parent),indent=2));return
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
        raise ValueError('Explicit deterministic CUBLAS workspace required before CUDA initialization')
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    rank=dist.get_rank();world=dist.get_world_size()
    from training.fsdp_checkpoint import save_checkpoint,load_checkpoint
    from training.official_fsdp import shard_official_qwen,reshard_official_parameters
    from training.flashoptim_convergence_job import Blocks
    error=[None]
    if rank==0:
        try:args.output.mkdir(parents=True,exist_ok=False)
        except Exception as exc:error[0]=str(exc)
    dist.broadcast_object_list(error,src=0)
    if error[0]:raise RuntimeError(error[0])
    try:
        if world!=cfg['world_size']:raise ValueError('Frozen protocol world size mismatch')
        def make():
            model=tiny_model().to(torch.bfloat16) if args.tiny else AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,
                torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True)
            keep=min(args.keep_full_layers,2) if args.tiny else args.keep_full_layers
            model=shard_official_qwen(model,keep_full_layers=keep,
                checkpoint_every=args.checkpoint_every).train()
            opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),
                eps=cfg['epsilon'],weight_decay=cfg['weight_decay'],fused=True)
            return model,opt
        model,optimizer=make();plan=resource_plan(model,args.output)
        sufficient=torch.tensor(int(plan['resources_sufficient']),device='cuda')
        dist.all_reduce(sufficient,op=dist.ReduceOp.MIN)
        if not sufficient.item():raise RuntimeError(f'Insufficient resources; no checkpoint written: {plan}')
        corpus=Blocks(Path(cfg['corpus'])/'train.bin',cfg['sequence_length'],training=True,seed=cfg['seed'])
        batch=cfg['microbatch_per_rank']
        torch.manual_seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);random.seed(cfg['seed']+rank)
        def update(m,opt,step):
            first=step*world*batch+rank*batch
            ids=torch.stack([corpus[first+j] for j in range(batch)])
            if args.tiny:ids=ids%97
            input_hash=hashlib.sha256(memoryview(ids.numpy())).hexdigest()
            opt.zero_grad(set_to_none=True)
            loss=m(ids.cuda(),labels=ids.cuda(),use_cache=False).loss
            loss.backward();opt.step();torch.cuda.synchronize()
            # Exercise all RNG streams without changing the training objective.
            random.random();np.random.random();torch.rand(());torch.rand((),device='cuda')
            return dict(loss=loss.item(),input_sha256=input_hash)
        initial=[update(model,optimizer,s) for s in range(2)]
        optimizer.zero_grad(set_to_none=True)
        metadata=dict(protocol=cfg,keep_full_layers=min(args.keep_full_layers,2) if args.tiny else args.keep_full_layers,
            checkpoint_every=args.checkpoint_every,deterministic=True,synthetic_tiny=args.tiny,
            test='real_resume_replay_v1')
        progress=dict(next_update=2,next_block=2*world*batch+rank*batch)
        save_checkpoint(args.output/'step2',model,optimizer,step=2,data_state=progress,metadata=metadata)
        expected=[update(model,optimizer,s) for s in range(2,4)]
        optimizer.zero_grad(set_to_none=True);reshard_official_parameters(model)
        expected_state=state_digest(model,optimizer);expected_rng=rng_samples()
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
        torch.manual_seed(999);np.random.seed(999);random.seed(999)
        model,optimizer=make()
        restored=load_checkpoint(args.output/'step2',model,optimizer,metadata=metadata)
        if restored!=dict(step=2,data_state=progress):raise AssertionError('Trainer/data progress mismatch')
        actual=[update(model,optimizer,s) for s in range(restored['data_state']['next_update'],4)]
        optimizer.zero_grad(set_to_none=True);reshard_official_parameters(model)
        actual_state=state_digest(model,optimizer);actual_rng=rng_samples()
        if actual!=expected or actual_state!=expected_state or actual_rng!=expected_rng:
            raise AssertionError('Real-model loss/input/state/RNG replay mismatch')
        report=dict(passed=True,rank=rank,world_size=world,resources=plan,initial_updates=initial,
            replay_updates=actual,state=actual_state,rng=actual_rng,deterministic=True,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Synthetic D128 pipeline gate; NOT real 8B proof' if args.tiny else
                'Real pretrained model two-step resume replay; not convergence or nondeterministic bitwise replay')
        (args.output/f'rank{rank}_result.json').write_text(json.dumps(report,indent=2)+'\n')
        dist.barrier()
        if rank==0:
            (args.output/'result.json').write_text(json.dumps(dict(passed=True,world_size=world,
                model='synthetic_tiny_qwen' if args.tiny else cfg['model'],synthetic_tiny=args.tiny,
                full_8b_resume_proven=not args.tiny and plan['parameters']==8190735360,replay_steps=2,
                parameter_optimizer_rng_replay='bitwise_equal_via_tensor_hashes',
                checkpoint_retained=str(args.output/'step2'),scope=report['scope']),indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
