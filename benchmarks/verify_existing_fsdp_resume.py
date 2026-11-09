"""Replay an existing step2 8B checkpoint without writing another large copy.

Rebuild the uninterrupted four-update reference from the frozen protocol, then
rebuild/load step2 and compare updates 3/4, master/moment hashes and RNG. Native
and offset-sorted readers are explicit choices; no in-flight monkeypatching.
"""
import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from benchmarks.verify_real_fsdp_resume import state_digest,rng_samples,resource_plan


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reader',choices=['native','offset_sorted'],default='native')
    p.add_argument('--protocol',type=Path,default=Path('training/fsdp_convergence_protocol_v1.json'))
    a=p.parse_args();cfg=json.loads(a.protocol.read_text())
    manifest=json.loads((a.checkpoint/'complete.json').read_text());metadata=manifest['metadata']
    if (manifest['step']!=2 or manifest['world_size']!=4 or metadata.get('protocol')!=cfg
            or metadata.get('synthetic_tiny') is not False or metadata.get('deterministic') is not True
            or metadata.get('test')!='real_resume_replay_v1'):
        raise ValueError('Matching real four-rank step2 checkpoint required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
        raise ValueError('Explicit deterministic CUBLAS workspace required')
    from transformers import AutoConfig,AutoModelForCausalLM
    # This is a read-only replay: require host staging RAM, but do NOT require
    # another 100.7 GiB of free disk or call save_checkpoint again.
    with torch.device('meta'):
        meta=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(cfg['model'],local_files_only=True))
    plan=resource_plan(meta,a.output.parent)
    del meta
    if plan['parameters']!=8190735360 or plan['host_ram_available_bytes']<plan['host_ram_required_bytes']:
        raise ValueError('Real 8B model and sufficient host staging RAM required')
    from training.fsdp_provenance import source_snapshot
    root=Path(__file__).resolve().parents[1];sources=source_snapshot(root)
    files=[Path(__file__).resolve(),root/'benchmarks/verify_real_fsdp_resume.py',root/'benchmarks/dcp_sorted_reader.py',
           root/'benchmarks/dcp_reader_adapter.py']
    def helper_hashes():return {str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    helpers=helper_hashes()
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(hours=1),
        device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    rank,world=dist.get_rank(),dist.get_world_size()
    try:
        if world!=4:raise ValueError('Four ranks required')
        error=[None]
        if rank==0:
            try:a.output.mkdir(parents=True,exist_ok=False)
            except Exception as exc:error[0]=str(exc)
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise RuntimeError(error[0])
        from training.official_fsdp import shard_official_qwen,reshard_official_parameters
        from training.fsdp_checkpoint import load_checkpoint
        from training.flashoptim_convergence_job import Blocks
        def make():
            model=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,
                torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True)
            model=shard_official_qwen(model,keep_full_layers=metadata['keep_full_layers'],
                checkpoint_every=metadata['checkpoint_every']).train()
            opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),
                eps=cfg['epsilon'],weight_decay=cfg['weight_decay'],fused=True)
            return model,opt
        model,opt=make()
        corpus=Blocks(Path(cfg['corpus'])/'train.bin',cfg['sequence_length'],training=True,seed=cfg['seed'])
        batch=cfg['microbatch_per_rank']
        torch.manual_seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);random.seed(cfg['seed']+rank)
        def update(m,optimizer,step):
            first=step*world*batch+rank*batch
            ids=torch.stack([corpus[first+j] for j in range(batch)])
            checksum=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            optimizer.zero_grad(set_to_none=True)
            loss=m(ids.cuda(),labels=ids.cuda(),use_cache=False).loss
            loss.backward();optimizer.step();torch.cuda.synchronize()
            random.random();np.random.random();torch.rand(());torch.rand((),device='cuda')
            return dict(loss=loss.item(),input_sha256=checksum)
        initial=[update(model,opt,s) for s in range(2)]
        opt.zero_grad(set_to_none=True);reshard_official_parameters(model)
        expected=[update(model,opt,s) for s in range(2,4)]
        opt.zero_grad(set_to_none=True);reshard_official_parameters(model)
        expected_state=state_digest(model,opt);expected_rng=rng_samples()
        del opt,model;gc.collect();torch.cuda.empty_cache()
        torch.manual_seed(999);np.random.seed(999);random.seed(999)
        model,opt=make()
        if a.reader=='offset_sorted':
            from benchmarks.dcp_reader_adapter import load_sorted_checkpoint
            loader=load_sorted_checkpoint
        else:loader=load_checkpoint
        tick=time.perf_counter()
        restored=loader(a.checkpoint,model,opt,metadata=metadata)
        load_seconds=time.perf_counter()-tick
        progress=dict(next_update=2,next_block=2*world*batch+rank*batch)
        if restored!=dict(step=2,data_state=progress):raise AssertionError('Trainer/data position mismatch')
        actual=[update(model,opt,s) for s in range(2,4)]
        opt.zero_grad(set_to_none=True);reshard_official_parameters(model)
        actual_state=state_digest(model,opt);actual_rng=rng_samples()
        if actual!=expected or actual_state!=expected_state or actual_rng!=expected_rng:
            raise AssertionError('Existing-checkpoint loss/input/state/RNG replay mismatch')
        if sources!=source_snapshot(root) or helpers!=helper_hashes():raise ValueError('Replay source changed')
        record=dict(passed=True,rank=rank,world_size=world,reader=a.reader,initial_updates=initial,
            expected_updates=expected,replay_updates=actual,state=actual_state,rng=actual_rng,
            load_seconds=load_seconds,checkpoint=str(a.checkpoint),wrote_checkpoint=False,
            deterministic=True,resources=dict(plan,read_only_replay_resources_sufficient=True),
            source_sha256=helpers['benchmarks/verify_existing_fsdp_resume.py'],
            implementation_sources=sources,helper_hashes=helpers,
            scope='Real 8B existing-checkpoint replay; timings are not a matched cold-cache I/O benchmark')
        (a.output/f'rank{rank}_result.json').write_text(json.dumps(record,indent=2))
        dist.barrier()
        if rank==0:
            (a.output/'result.json').write_text(json.dumps(dict(passed=True,world_size=world,
                full_8b_resume_proven=True,replay_steps=2,reader=a.reader,wrote_checkpoint=False,
                synthetic_tiny=False,model=cfg['model'],checkpoint_retained=str(a.checkpoint),
                parameter_optimizer_rng_replay='bitwise_equal_via_tensor_hashes',
                checkpoint=str(a.checkpoint),scope=record['scope']),indent=2))
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
