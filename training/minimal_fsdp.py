"""Educational single-unit full sharding; NCCL communication, Triton local ops.

Whole unit is recomputed in backward to release gathered parameters after forward.
Not production FSDP: one Tensor input/output, no buffers, tied weights, RNG,
hooks, accumulation, higher-order gradients, AMP or checkpoint/export support.
All ranks must execute identical collective order. Inputs must not be mutated.
"""
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.func import functional_call


@triton.jit
def _copy_scale(src, dst, size, scale, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + i, i < size, other=0).to(tl.float32)
    tl.store(dst + i, value * scale, i < size)


def copy_scale(src, scale=1., *, dtype=None):
    out = torch.empty_like(src, dtype=dtype or src.dtype)
    _copy_scale[(triton.cdiv(src.numel(), 256),)](src, out, src.numel(), scale, BLOCK=256)
    return out


class _ShardedCall(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, x, owner):
        ctx.owner = owner
        ctx.save_for_backward(shard, x)
        full = owner.gather(shard)
        # Custom Function forward is no-grad: no full weights/activations retained.
        return owner.call(full, x)

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError('First-order backward only')
        owner = ctx.owner
        shard, x = ctx.saved_tensors
        with torch.enable_grad():
            full = owner.gather(shard).requires_grad_(True)
            local_x = x.detach().requires_grad_(True)
            output = owner.call(full, local_x)
            grad_full, grad_x = torch.autograd.grad(output, (full, local_x), grad_output)
        local = torch.empty_like(shard)
        dist.reduce_scatter_tensor(local, grad_full.contiguous(), op=dist.ReduceOp.SUM)
        # Equal local batch sizes: average parameter gradients over ranks.
        return copy_scale(local, 1. / owner.world), grad_x, None


class MinimalFSDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        if not dist.is_initialized() or dist.get_backend() != 'nccl':
            raise ValueError('Initialize an NCCL process group first')
        if list(module.buffers()):
            raise ValueError('Buffers are not supported in this prototype')
        if any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p > 0
               for m in module.modules()):
            raise ValueError('Dropout needs RNG-preserving recomputation, not supported yet')
        entries = list(module.named_parameters(remove_duplicate=False))
        if not entries or len({id(p) for _, p in entries}) != len(entries):
            raise ValueError('Nonempty, untied parameters required')
        params = [p for _, p in entries]
        if any(not p.is_cuda or p.dtype != torch.float32 or not p.requires_grad for p in params):
            raise ValueError('Trainable CUDA FP32 parameters required')
        if any(p.device != params[0].device for p in params):
            raise ValueError('All parameters must share one device')
        self.world, self.rank = dist.get_world_size(), dist.get_rank()
        self.spec = [(name, p.shape, p.numel()) for name, p in entries]
        self.numel = sum(p.numel() for p in params)
        self.shard_size = triton.cdiv(self.numel, self.world)
        flat = torch.zeros(self.shard_size * self.world, device=params[0].device)
        with torch.no_grad():
            flat[:self.numel].copy_(torch.cat([p.detach().reshape(-1) for p in params]))
            dist.broadcast(flat, src=0)
        start = self.rank * self.shard_size
        self.shard = torch.nn.Parameter(copy_scale(flat[start:start + self.shard_size]))
        # Keep architecture only, not full replicated parameter storage.
        module.to('meta')
        object.__setattr__(self, '_template', module)

    def gather(self, shard):
        full = torch.empty(self.shard_size * self.world, device=shard.device, dtype=shard.dtype)
        dist.all_gather_into_tensor(full, shard.contiguous())
        return full

    def call(self, full, x):
        state, offset = {}, 0
        for name, shape, count in self.spec:
            state[name] = full[offset:offset + count].view(shape)
            offset += count
        return functional_call(self._template, state, (x,), strict=True)

    def forward(self, x):
        if x.device != self.shard.device or x.dtype != torch.float32:
            raise ValueError('Input must be FP32 on the local shard device')
        return _ShardedCall.apply(self.shard, x, self)
