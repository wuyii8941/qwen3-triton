"""Private-version, destructive-scratch FSDP experiment. Never a default.

Only the isolated gate injects this adapter. Ownership is restricted to the
exact tensor returned by this object's allocate(), not caller-provided views.
"""
import hashlib
from pathlib import Path
import weakref

import torch
import torch.distributed as dist
import triton

from benchmarks.bf16_exchange_reduce import _sum_contributions
from benchmarks.bf16_scratch_pack import pack_owned_scratch


PINNED = {
    '_fsdp_collectives.py': 'f8527a45d7d8ae029e8b2ce78b9fd6fc909b95e4b3cf1be7cfa9ba3da722be31',
    '_fsdp_param_group.py': '0f81daf147e30914f9f96bd5e654c3bc3df7a5f4a0db37819ed76d3f00858cb6',
}


def verify_private_contract():
    if torch.__version__ != '2.10.0+cu128':
        raise ValueError('Scratch mutation is pinned to tested Torch2.10.0+cu128')
    root = Path(torch.__file__).parent / 'distributed/fsdp/_fully_shard'
    for name, digest in PINNED.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Untested FSDP scratch lifetime source: {name}')


class ScratchExchangeReduceScatter:
    # Gate keeps these instances solely to prove calls on every rank.
    instances = []

    def __init__(self, *, original_bf16, accumulation=1, backend='triton', debug=True):
        if original_bf16 is not True or accumulation != 1 or backend != 'triton' or debug is not True:
            raise ValueError('Only original BF16, accumulation1, Triton, debug gate supported')
        verify_private_contract()
        self.calls = 0
        self.owned = weakref.WeakValueDictionary()
        self.instances.append(self)

    def allocate(self, size, *, dtype, device):
        if dtype != torch.float32 or torch.device(device).type != 'cuda':
            raise ValueError('Only FP32 CUDA scratch is supported')
        tensor = torch.empty(size, dtype=dtype, device=device)
        self.owned[id(tensor)] = tensor
        return tensor

    @torch.no_grad()
    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        if async_op or op not in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
            raise ValueError('Only synchronous enqueue SUM/AVG supported; no custom divisor')
        world = dist.get_world_size(group)
        valid = (world >= 2 and self.owned.get(id(input_tensor)) is input_tensor
                 and self.owned.get(id(output_tensor)) is output_tensor
                 and input_tensor.dtype == output_tensor.dtype == torch.float32
                 and input_tensor.is_cuda and output_tensor.device == input_tensor.device
                 and input_tensor.ndim == output_tensor.ndim == 1
                 and input_tensor.is_contiguous() and output_tensor.is_contiguous()
                 and input_tensor._base is None and input_tensor.storage_offset() == 0
                 and not input_tensor.requires_grad
                 and input_tensor.untyped_storage().nbytes() == input_tensor.numel() * 4
                 and input_tensor.numel() == output_tensor.numel() * world
                 and output_tensor.numel() > 0)
        if valid:
            valid = (torch.isfinite(input_tensor).all().item()
                     and not ((input_tensor.view(torch.int32) & 65535) != 0).any().item())
        # Reject collectively, before mutating or entering all-to-all.
        flag = torch.tensor(int(valid), device=input_tensor.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        if not flag.item():
            raise ValueError('Every rank must supply owned finite exact-BF16 FP32 scratch')
        send, receive = pack_owned_scratch(input_tensor, allow_mutation=True, backend='triton')
        dist.all_to_all_single(receive, send, group=group)
        cols = output_tensor.numel()
        _sum_contributions[(triton.cdiv(cols, 1024),)](
            receive, output_tensor, cols, world, 1 / world if op == dist.ReduceOp.AVG else 1.,
            triton.next_power_of_2(world), 1024, num_warps=4)
        self.calls += 1
