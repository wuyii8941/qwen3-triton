"""Experimental reader injection with the native checkpoint validation contract.

Kept outside training to avoid changing the live native 8B replay. No global
monkeypatches: only the explicit dcp.load call receives a storage reader.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions,get_state_dict,set_state_dict

from benchmarks.dcp_sorted_reader import OffsetSortedFileSystemReader
from training.fsdp_checkpoint import _canonical,_rank0_check,_restore_rng,schema


def load_sorted_checkpoint(path,model,optimizer,*,metadata):
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
    dcp.load(payload,storage_reader=OffsetSortedFileSystemReader(path/'state'))
    set_state_dict(model,optimizer,model_state_dict=payload['model'],
        optim_state_dict=payload['optimizer'],options=options)
    trainer=torch.load(path/f'rank{dist.get_rank()}_trainer.pt',map_location='cpu',weights_only=True)
    if trainer['step']!=manifest['step']:raise ValueError('Trainer step mismatch')
    _restore_rng(trainer)
    return dict(step=trainer['step'],data_state=trainer['data_state'])
