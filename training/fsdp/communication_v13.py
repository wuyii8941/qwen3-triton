"""Opt-in FP32 FSDP communication experiments for four ordered local GPUs.

Uses PyTorch 2.10's public custom-RS setter with its version-specific Comm
interface. No monkeypatching of PyTorch. Arithmetic stays FP32 and AVG/SUM
normalization is preserved. The hierarchical path changes reduction order.
Rank pairs (0,1)/(2,3) must match the measured machine topology.
"""
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatter
import triton
import triton.language as tl


@triton.jit
def _pair_pack(X, Y, N: tl.constexpr, Q: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    chunk = i // Q
    source = ((chunk % 2) * 2 + chunk // 2) * Q + i % Q
    value = tl.load(X + source, i < N, 0)
    tl.store(Y + i, value, i < N)


class ExperimentalReduceScatter(ReduceScatter):
    def __init__(self, mode, *, minimum_bytes=64 * 2**20):
        if torch.__version__.split('+')[0] != '2.10.0':
            raise RuntimeError('This experimental Comm interface is pinned to Torch 2.10.0')
        if mode not in ('separate', 'hierarchical'):
            raise ValueError('Expected separate or hierarchical')
        if dist.get_world_size() != 4:
            raise ValueError('Four ranks required')
        self.mode = mode
        self.minimum_bytes = minimum_bytes
        self.calls = dict(native=0, separate=0, hierarchical=0)
        self.max_input_bytes = 0
        self.groups = []
        self.pair = self.cross = self.separate = None
        rank = dist.get_rank()
        layouts = [[0,1,2,3]] if mode == 'separate' else [[0,1],[2,3],[0,2],[1,3]]
        for members in layouts:
            group = dist.new_group(members, backend='nccl', timeout=timedelta(minutes=10),
                                   device_id=torch.device('cuda', torch.cuda.current_device()))
            if rank in members:
                self.groups.append(group)
                if mode == 'separate': self.separate = group
                elif members in [[0,1],[2,3]]: self.pair = group
                else: self.cross = group

    def allocate(self, size, *, dtype, device):
        return torch.empty(tuple(size), dtype=dtype, device=device)

    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        if dist.get_process_group_ranks(group) != [0,1,2,3]:
            raise ValueError('Expected global rank order 0,1,2,3')
        if input_tensor.dtype != torch.float32 or output_tensor.dtype != torch.float32:
            raise TypeError('FP32 communication required')
        if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
            raise ValueError('Contiguous communication buffers required')
        n = input_tensor.numel()
        if n != 4 * output_tensor.numel(): raise ValueError('Invalid RS sizes')
        self.max_input_bytes = max(self.max_input_bytes, n * 4)
        if self.mode == 'separate':
            self.calls['separate'] += 1
            return dist.reduce_scatter_tensor(output_tensor, input_tensor, group=self.separate,
                                              op=op, async_op=async_op)
        if n * 4 < self.minimum_bytes:
            self.calls['native'] += 1
            return dist.reduce_scatter_tensor(output_tensor, input_tensor, group=group,
                                              op=op, async_op=async_op)
        if async_op or op not in (dist.ReduceOp.AVG, dist.ReduceOp.SUM):
            raise ValueError('Hierarchical path supports synchronous AVG/SUM only')
        packed = torch.empty_like(input_tensor)
        half = torch.empty(n // 2, dtype=torch.float32, device=input_tensor.device)
        _pair_pack[(triton.cdiv(n, 1024),)](input_tensor, packed, n, n // 4, 1024)
        dist.reduce_scatter_tensor(half, packed, group=self.pair, op=op)
        result = dist.reduce_scatter_tensor(output_tensor, half, group=self.cross, op=op)
        self.calls['hierarchical'] += 1
        return result

    def report(self):
        return dict(mode=self.mode, calls=dict(self.calls), minimum_bytes=self.minimum_bytes,
                    max_input_bytes=self.max_input_bytes, gradient_communication='FP32',
                    hierarchy=[[0,1],[2,3],[0,2],[1,3]] if self.mode=='hierarchical' else None)


def configure_communication(model, mode, *, minimum_bytes=64 * 2**20):
    if mode == 'native': return None
    comm = ExperimentalReduceScatter(mode, minimum_bytes=minimum_bytes)
    units = 0
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_custom_reduce_scatter(comm)
            units += 1
    if not units: raise ValueError('No FSDP units found')
    return comm
