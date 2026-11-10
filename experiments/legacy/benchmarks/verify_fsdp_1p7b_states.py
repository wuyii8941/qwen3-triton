"""Real1.7B deterministic full-local-shard gradient/master/AdamW byte gate.

Streams one tensor at a time to CPU and keeps only hashes, never checkpoints
or complete reference tensors. All ranks compare all named parameters.
"""
import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoModelForCausalLM

from training.fsdp_activation import canonical_parameter_name
from training.fsdp_provenance import source_snapshot
from training.official_fsdp import shard_official_qwen,reshard_official_parameters
from training.flashoptim_convergence_job import Blocks


def tensor_record(tensor):
    local=tensor.to_local() if hasattr(tensor,'to_local') else tensor
    host=local.detach().cpu().contiguous()
    return dict(shape=list(tensor.shape),local_shape=list(local.shape),dtype=str(local.dtype),
        finite=bool(torch.isfinite(host).all()),
        sha256=hashlib.sha256(memoryview(host.reshape(-1).view(torch.uint8).numpy())).hexdigest())


def named_parameters(model):
    pairs=[(canonical_parameter_name(n),p) for n,p in model.named_parameters()]
    if len({n for n,p in pairs})!=len(pairs):raise ValueError('Canonical parameter collision')
    return dict(pairs)


def host_available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:')))*1024


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Explicit deterministic workspace required')
    root=Path.cwd();sources=source_snapshot(root)
    own=Path(__file__).resolve();own_hash=hashlib.sha256(own.read_bytes()).hexdigest()
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    total=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(24*2**30/total)
    dist.init_process_group('nccl',timeout=timedelta(minutes=5),
        device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    rank,world=dist.get_rank(),dist.get_world_size()
    try:
        if world!=2:raise ValueError('Two-rank diagnostic required')
        if rank==0:args.output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        corpus=Blocks('artifacts/convergence_corpus_v1/train.bin',512,training=True,seed=20261106)
        inputs=[corpus[step*world+rank].unsqueeze(0).cuda() for step in range(3)]
        reference=[];reports={}
        for label,keep,every in (('native_a',0,0),('native_b',0,0),('keep28',28,0),('checkpoint28',28,1)):
            ready=torch.tensor(int(host_available()>=16*2**30),device='cuda')
            dist.all_reduce(ready,op=dist.ReduceOp.MIN)
            if not ready.item():raise RuntimeError('Insufficient 16GiB host staging margin')
            torch.manual_seed(20261106)
            model=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-1.7B',
                local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True)
            model=shard_official_qwen(model,keep_full_layers=keep,checkpoint_every=every).train()
            params=named_parameters(model)
            if len(params)!=310 or sum(p.numel() for p in params.values())!=1720574976:
                raise ValueError('Wrong real1.7B parameter coverage')
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,
                weight_decay=.01,fused=True)
            records=[]
            for step,ids in enumerate(inputs):
                optimizer.zero_grad(set_to_none=True)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    loss=model(ids,labels=ids,use_cache=False).loss
                    loss.backward()
                grads={n:tensor_record(p.grad) for n,p in params.items()}
                optimizer.step();torch.cuda.synchronize();reshard_official_parameters(model)
                masters={n:tensor_record(p) for n,p in params.items()}
                states={n:{k:tensor_record(v) if isinstance(v,torch.Tensor) else v
                    for k,v in optimizer.state[p].items()} for n,p in params.items()}
                buffers={canonical_parameter_name(n):tensor_record(v) for n,v in model.named_buffers()}
                groups=[{k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups]
                record=dict(step=step+1,input_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                    loss=loss.item(),gradients=grads,parameters=masters,optimizer_states=states,
                    buffers=buffers,param_groups=groups)
                tensors=list(grads.values())+list(masters.values())+list(buffers.values())+[
                    v for state in states.values() for v in state.values() if isinstance(v,dict)]
                finite=all(t['finite'] for t in tensors) and bool(torch.isfinite(loss).item())
                if any(t['dtype']!='torch.float32' for t in list(grads.values())+list(masters.values())):
                    raise ValueError('Expected FP32 master/gradient shards')
                if any(set(s)!={'step','exp_avg','exp_avg_sq'} for s in states.values()):
                    raise ValueError('All AdamW states required')
                matched=finite and (label=='native_a' or record==reference[step])
                records.append(record)
                if label=='native_a':reference.append(record)
                all_ok=torch.tensor(int(matched),device='cuda');dist.all_reduce(all_ok,op=dist.ReduceOp.MIN)
                if not all_ok.item():
                    (args.output/f'rank{rank}_{label}_failure.json').write_text(json.dumps(dict(
                        phase=label,step=step+1,finite=finite,actual=record,
                        expected=reference[step]),indent=2))
                    raise AssertionError(f'{label} step{step+1}: full gradient/state byte comparison failed')
                if rank==0:print(json.dumps(dict(phase=label,step=step+1,passed=True,
                    parameters_per_rank=310,gradient_tensors_per_rank=310)),flush=True)
            reports[label]=records
            del params,optimizer,model,loss
            gc.collect();torch.cuda.empty_cache();dist.barrier()
        current=source_snapshot(root)==sources and hashlib.sha256(own.read_bytes()).hexdigest()==own_hash
        valid=torch.tensor(int(current),device='cuda');dist.all_reduce(valid,op=dist.ReduceOp.MIN)
        if not valid.item():raise ValueError('Source changed during gate')
        result=dict(passed=True,rank=rank,world=world,deterministic=True,real_parameters=1720574976,
            parameter_entries_per_rank=310,steps_per_phase=3,phases=reports,
            implementation_sources=sources,worker_sha256=own_hash,
            scope='All local shards/all ranks: finite gradients, FP32 masters, AdamW state and buffers exact by full-byte SHA256 for three updates. Not convergence, performance or arbitrary hardware proof')
        (args.output/f'rank{rank}_result.json').write_text(json.dumps(result,indent=2))
        dist.barrier()
        if rank==0:
            (args.output/'result.json').write_text(json.dumps(dict(passed=True,world=world,
                real_parameters=1720574976,parameter_entries_per_rank=310,steps_per_phase=3,
                phases=list(reports),implementation_sources=sources,worker_sha256=own_hash,
                scope=result['scope']),indent=2))
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
