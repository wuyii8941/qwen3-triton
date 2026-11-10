"""Collect one unit at a time into a reloadable BF16 Hugging Face snapshot.

All ranks participate. Only rank 0 writes files. This is a model export, NOT
an optimizer checkpoint, and cannot exactly resume FP32-master AdamW training.
"""
import copy
import json
import shutil
from pathlib import Path
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from training.block_fsdp import BlockFSDP


@torch.no_grad()
def export_hf(model, destination, source):
    destination, source = Path(destination), Path(source)
    units = [(name,unit) for name,unit in model.named_modules() if isinstance(unit,BlockFSDP)]
    total_bytes = sum(unit.numel*2 for _,unit in units)
    status = [None]
    if dist.get_rank() == 0:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists(): status[0] = 'Export directory already exists; refusing overwrite'
        elif shutil.disk_usage(destination.parent).free < total_bytes*1.1:
            status[0] = 'Not enough disk space for BF16 model export'
        else: destination.mkdir()
    dist.broadcast_object_list(status,src=0)
    if status[0]: raise RuntimeError(status[0])
    dist.barrier()
    weight_map = {}
    for index,(prefix,unit) in enumerate(units):
        old_dtype = unit.compute_dtype
        unit.compute_dtype = torch.bfloat16
        full = unit.gather(unit.shard.detach())
        unit.compute_dtype = old_dtype
        error = [None]
        if dist.get_rank() == 0:
            try:
                flat = full[:unit.numel].cpu()
                state,offset = {},0
                name = f'model-{index+1:05d}-of-{len(units):05d}.safetensors'
                for key,shape,count in unit.spec:
                    qualified = f'{prefix}.{key}'
                    state[qualified] = flat[offset:offset+count].view(shape).clone()
                    weight_map[qualified] = name
                    offset += count
                save_file(state,str(destination/name),metadata={'format':'pt'})
                del flat,state
            except Exception as exc:
                error[0] = str(exc)
        del full
        dist.broadcast_object_list(error,src=0)
        if error[0]: raise RuntimeError(f'Export failed: {error[0]}')
    if dist.get_rank() == 0:
        config = copy.deepcopy(model.config)
        config.use_cache = True
        config.torch_dtype = torch.bfloat16
        config.save_pretrained(destination)
        if getattr(model,'generation_config',None): model.generation_config.save_pretrained(destination)
        for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                     'added_tokens.json','vocab.json','merges.txt','chat_template.jinja','LICENSE','LICENSE.txt'):
            if (source/name).is_file(): shutil.copyfile(source/name,destination/name)
        (destination/'model.safetensors.index.json').write_text(json.dumps(dict(
            metadata=dict(total_size=total_bytes),weight_map=weight_map),indent=2))
        (destination/'export_complete.json').write_text(json.dumps(dict(
            source=str(source),world_size=dist.get_world_size(),dtype='bfloat16',
            optimizer_state=False,total_bytes=total_bytes),indent=2))
    dist.barrier()
