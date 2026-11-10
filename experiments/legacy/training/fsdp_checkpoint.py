"""Same-world FP32 AdamW DCP checkpoint + per-rank trainer/RNG sidecars.

Official FSDP2 only (or replicated CPU models for independent tests). Call at
an optimizer-step boundary. This is synchronous and never consolidates full
GPU parameters. Disk guard runs before CPU state materialization.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions,get_state_dict,set_state_dict
from training.official_fsdp import reshard_official_parameters


def schema(model):
    return [[n,list(p.shape),str(p.dtype)] for n,p in model.named_parameters()]


def required_bytes(model):
    # FP32 master + two FP32 AdamW moments; no gradients saved. DTensor.numel
    # describes the global tensor, so do not multiply by world size again.
    size=sum(p.numel()*12 for p in model.parameters())
    size+=sum(b.numel()*b.element_size() for b in model.buffers())
    return int(size*1.1)+16*2**20


def _canonical(model,optimizer):
    if type(optimizer) is not torch.optim.AdamW:raise ValueError('Standard FP32 AdamW required')
    if any(g.get('amsgrad',False) for g in optimizer.param_groups):
        raise ValueError('AMSGrad has additional state not covered by this disk budget')
    from training.block_fsdp import BlockFSDP
    if any(isinstance(m,BlockFSDP) for m in model.modules()):
        raise ValueError('Use custom BlockFSDP checkpoint implementation instead')
    reshard_official_parameters(model)
    if any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Canonical FP32 master parameters required')
    for state in optimizer.state.values():
        if any(state[k].dtype!=torch.float32 for k in ('exp_avg','exp_avg_sq') if k in state):
            raise ValueError('FP32 optimizer moments required')


def _rank0_check(check):
    message=[None]
    if dist.get_rank()==0:
        try:message[0]=check()
        except Exception as error:message[0]=f'{type(error).__name__}: {error}'
    dist.broadcast_object_list(message,src=0)
    if message[0]:raise RuntimeError(message[0])


def _trainer_state(step,data_state,uses_cuda):
    numpy=np.random.get_state()
    return dict(step=step,data_state=copy.deepcopy(data_state),cpu_rng=torch.get_rng_state(),
        python_rng=random.getstate(),numpy_rng=dict(name=numpy[0],
            keys=torch.from_numpy(numpy[1].astype(np.int64)),position=numpy[2],
            has_gauss=numpy[3],cached_gauss=numpy[4]),
        cuda_rng=torch.cuda.get_rng_state() if uses_cuda else None)


def _restore_rng(trainer):
    numpy=trainer['numpy_rng']
    random.setstate(trainer['python_rng'])
    np.random.set_state((numpy['name'],numpy['keys'].numpy().astype(np.uint32),numpy['position'],
        numpy['has_gauss'],numpy['cached_gauss']))
    torch.set_rng_state(trainer['cpu_rng'])
    if trainer['cuda_rng'] is not None:
        if not torch.cuda.is_initialized():raise ValueError('CUDA checkpoint requires CUDA model initialization')
        torch.cuda.set_rng_state(trainer['cuda_rng'])


def save_checkpoint(path,model,optimizer,*,step,data_state,metadata):
    if not dist.is_initialized():raise ValueError('Initialized process group required')
    if not isinstance(step,int) or step<1 or not optimizer.state:
        raise ValueError('Save after an initialized optimizer step, not before training')
    json.dumps(data_state,allow_nan=False);json.dumps(metadata,allow_nan=False)
    _canonical(model,optimizer)
    path=Path(path)
    uses_cuda=any(p.is_cuda for p in model.parameters())
    manifest=dict(schema_version=1,world_size=dist.get_world_size(),step=step,
        metadata=metadata,model_schema=schema(model),optimizer='FP32 AdamW',torch=str(torch.__version__),
        backend=str(dist.get_backend()),cuda_rng=uses_cuda)
    def reserve():
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():return 'Checkpoint destination already exists; refuse overwrite'
        required=required_bytes(model)
        if shutil.disk_usage(path.parent).free<required:
            return f'Insufficient disk: need at least {required/2**30:.2f} GiB; no checkpoint written'
        path.mkdir()
    _rank0_check(reserve)
    dist.barrier()
    # Capture before checkpoint APIs can initialize or touch internal state.
    trainer=_trainer_state(step,data_state,uses_cuda)
    options=StateDictOptions(full_state_dict=False,cpu_offload=True)
    model_state,optimizer_state=get_state_dict(model,optimizer,options=options)
    dcp.save(dict(model=model_state,optimizer=optimizer_state),checkpoint_id=path/'state')
    sidecar=path/f'rank{dist.get_rank()}_trainer.pt'
    temporary=sidecar.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(trainer,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,sidecar)
    dist.barrier()
    def commit():
        manifest['trainer_sha256']={str(rank):hashlib.sha256((path/f'rank{rank}_trainer.pt').read_bytes()).hexdigest()
            for rank in range(dist.get_world_size())}
        temporary=path/'complete.tmp'
        with temporary.open('w') as stream:
            json.dump(manifest,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path/'complete.json')
        directory=os.open(path,os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    _rank0_check(commit)
    dist.barrier()
    _restore_rng(trainer)


def load_checkpoint(path,model,optimizer,*,metadata):
    if not dist.is_initialized():raise ValueError('Initialized process group required')
    _canonical(model,optimizer);path=Path(path)
    def validate():
        if not (path/'complete.json').is_file():return 'Incomplete checkpoint: completion marker missing'
        m=json.loads((path/'complete.json').read_text())
        if m.get('schema_version')!=1 or m['world_size']!=dist.get_world_size():return 'Checkpoint world/schema mismatch'
        if m.get('backend')!=str(dist.get_backend()) or m.get('cuda_rng')!=any(p.is_cuda for p in model.parameters()):
            return 'Checkpoint backend/RNG topology mismatch'
        if m['metadata']!=metadata or m['model_schema']!=schema(model):return 'Checkpoint protocol/model mismatch'
        for rank in range(dist.get_world_size()):
            sidecar=path/f'rank{rank}_trainer.pt'
            if hashlib.sha256(sidecar.read_bytes()).hexdigest()!=m['trainer_sha256'][str(rank)]:
                return 'Trainer state checksum mismatch'
    _rank0_check(validate)
    manifest=json.loads((path/'complete.json').read_text())
    options=StateDictOptions(full_state_dict=False,cpu_offload=True)
    ms,os_=get_state_dict(model,optimizer,options=options)
    payload=dict(model=ms,optimizer=os_)
    dcp.load(payload,checkpoint_id=path/'state')
    set_state_dict(model,optimizer,model_state_dict=payload['model'],
        optim_state_dict=payload['optimizer'],options=options)
    trainer=torch.load(path/f'rank{dist.get_rank()}_trainer.pt',map_location='cpu',weights_only=True)
    if trainer['step']!=manifest['step']:raise ValueError('Trainer step mismatch')
    _restore_rng(trainer)
    return dict(step=trainer['step'],data_state=trainer['data_state'])
