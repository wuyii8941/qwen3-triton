"""Same-world-size sharded checkpoints; no full parameter consolidation."""
import json
import os
import shutil
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map
from training.block_fsdp import BlockFSDP


def schema(model):
    return [(name,unit.numel,unit.size,str(unit.compute_dtype))
            for name,unit in model.named_modules() if isinstance(unit,BlockFSDP)]


def save_sharded(path, model, optimizer, step, metadata):
    path = Path(path)
    # Check before any write; refuse old checkpoints instead of overwriting.
    count = sum(p.numel() for p in model.parameters())
    required = count*12*dist.get_world_size()*1.1
    status = [None]
    if dist.get_rank() == 0:
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): status[0] = 'Checkpoint destination already exists'
        elif shutil.disk_usage(path.parent).free < required:
            status[0] = f'Insufficient disk: need approximately {required/2**30:.1f} GiB'
        else: path.mkdir()
    dist.broadcast_object_list(status,src=0)
    if status[0]: raise RuntimeError(status[0])
    dist.barrier()
    payload = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
        world=dist.get_world_size(),schema=schema(model),metadata=metadata,
        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    payload = tree_map(lambda x:x.detach().cpu() if isinstance(x,torch.Tensor) else x,payload)
    target = path/f'rank{dist.get_rank()}.pt'
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,target)
    dist.barrier()
    if dist.get_rank() == 0:
        (path/'complete.json').write_text(json.dumps(dict(world=dist.get_world_size(),step=step)))
    dist.barrier()


def load_sharded(path, model, optimizer, metadata):
    path = Path(path)
    complete = json.loads((path/'complete.json').read_text())
    if complete['world'] != dist.get_world_size(): raise ValueError('World size mismatch')
    saved = torch.load(path/f'rank{dist.get_rank()}.pt',map_location='cpu',weights_only=False)
    if saved['schema'] != schema(model) or saved['metadata'] != metadata:
        raise ValueError('Checkpoint architecture/precision/training protocol mismatch')
    model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
    return saved['step']
