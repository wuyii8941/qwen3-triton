"""Checked scratch packing without full-size validation tensors.

Experimental, not used by the active real8B study or default training path.
Ownership remains a caller obligation; finite-value checks do not prove it.
"""
import sys

import torch
import torch.distributed as dist
import triton

from benchmarks.bf16_scratch_pack import _compact_stage, packing_stages
from benchmarks.bf16_scratch_validation import validate_upcast_flag
from benchmarks.fsdp_scratch_exchange_adapter import ScratchExchangeReduceScatter
from benchmarks.bf16_exchange_reduce import _sum_contributions


@torch.no_grad()
def pack_owned_scratch_bounded(scratch, *, allow_mutation=False, group=None):
    """Validate then destructively pack; group enables collective rejection.

    Every rank must provide a CUDA tensor on its rank-local collective device.
    Given that precondition, layout/value failures are agreed before any write.
    Host synchronization is explicit and included in any future benchmark.
    """
    if allow_mutation is not True:
        raise ValueError('Explicit scratch ownership and mutation permission required')
    if not scratch.is_cuda:
        raise ValueError('Rank-local CUDA tensor required')
    metadata = (sys.byteorder == 'little' and scratch.dtype == torch.float32
                and scratch.ndim == 1 and scratch.is_contiguous() and scratch.numel() > 0
                and scratch.storage_offset() == 0 and scratch._base is None
                and not scratch.requires_grad
                and scratch.untyped_storage().nbytes() == scratch.numel() * 4)
    flag = (validate_upcast_flag(scratch) if metadata else
            torch.zeros((), dtype=torch.int32, device=scratch.device))
    if group is not None:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    if not flag.item():
        raise ValueError('Owned standalone finite exact-BF16 FP32 scratch required on every rank')
    words, count = scratch.view(torch.bfloat16), scratch.numel()
    for lo, hi in packing_stages(count):
        _compact_stage[(triton.cdiv(hi - lo, 1024),)](words, lo, hi, BLOCK=1024)
    return words[:count], words[count:]


class BoundedScratchExchangeReduceScatter(ScratchExchangeReduceScatter):
    """Gate-only adapter: pinned private lifetime, full checks, bounded temp."""

    @torch.no_grad()
    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        if async_op or op not in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
            raise ValueError('Only SUM/AVG enqueue supported')
        if not input_tensor.is_cuda:
            raise ValueError('Rank-local CUDA input precondition required')
        world = dist.get_world_size(group)
        valid = (world >= 2 and self.owned.get(id(input_tensor)) is input_tensor
                 and self.owned.get(id(output_tensor)) is output_tensor
                 and input_tensor.dtype == output_tensor.dtype == torch.float32
                 and input_tensor.device == output_tensor.device
                 and input_tensor.ndim == output_tensor.ndim == 1
                 and input_tensor.is_contiguous() and output_tensor.is_contiguous()
                 and input_tensor.numel() == output_tensor.numel() * world
                 and output_tensor.numel() > 0)
        flag = torch.tensor(int(valid), dtype=torch.int32, device=input_tensor.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        if not flag.item():
            raise ValueError('All ranks must provide matching adapter-owned input/output')
        send, receive = pack_owned_scratch_bounded(input_tensor, allow_mutation=True, group=group)
        dist.all_to_all_single(receive, send, group=group)
        cols = output_tensor.numel()
        _sum_contributions[(triton.cdiv(cols, 1024),)](
            receive, output_tensor, cols, world, 1 / world if op == dist.ReduceOp.AVG else 1.,
            triton.next_power_of_2(world), 1024, num_warps=4)
        self.calls += 1
