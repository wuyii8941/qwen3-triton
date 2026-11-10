"""Synchronous block full-sharding with FP32 master shards and BF16 compute.

NCCL collectives, Triton local casts/scaling, recomputation autograd. Designed
for deterministic Qwen blocks (dropout=0); no overlap or higher-order gradients.
"""
import torch
import torch.distributed as dist
from torch.func import functional_call
from torch.utils._pytree import tree_flatten, tree_unflatten
from training.fsdp.primitives import copy_scale


class _Call(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, structure, constants, shard, *inputs):
        ctx.owner, ctx.structure, ctx.constants = owner, structure, constants
        ctx.save_for_backward(shard, *inputs)
        args, kwargs = owner.unpack(structure, constants, inputs)
        output = owner.call(owner.gather(shard), args, kwargs)
        # Qwen decoder returns (hidden_states,) when cache/attentions are off.
        ctx.tuple_output = isinstance(output, tuple)
        if ctx.tuple_output:
            if len(output) != 1:
                raise ValueError('Only single-Tensor decoder output supported')
            output = output[0]
        return output

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError('First-order backward only')
        shard, *saved = ctx.saved_tensors
        owner = ctx.owner
        with torch.enable_grad():
            full = owner.gather(shard).requires_grad_(True)
            inputs = [t.detach().requires_grad_(t.requires_grad) for t in saved]
            differentiable = [t for t in inputs if t.requires_grad]
            args, kwargs = owner.unpack(ctx.structure, ctx.constants, inputs)
            output = owner.call(full, args, kwargs)
            if ctx.tuple_output:
                output = output[0]
            gradients = torch.autograd.grad(output, (full, *differentiable), grad)
        # FP32 gradient communication and persistent optimizer/master state.
        grad_full = (gradients[0].contiguous() if gradients[0].dtype == torch.float32
                     else copy_scale(gradients[0].contiguous(),dtype=torch.float32))
        local = torch.empty_like(shard)
        dist.reduce_scatter_tensor(local, grad_full,
            op=dist.ReduceOp.AVG if owner.reduce_average else dist.ReduceOp.SUM)
        iterator = iter(gradients[1:])
        input_grads = [next(iterator) if t.requires_grad else None for t in inputs]
        return (None, None, None, local if owner.reduce_average else copy_scale(local, 1. / owner.world), *input_grads)


class BlockFSDP(torch.nn.Module):
    def __init__(self, module, *, shared=None, compute_dtype=torch.bfloat16):
        super().__init__()
        if dist.get_backend() != 'nccl':
            raise ValueError('NCCL process group required')
        if list(module.buffers()):
            raise ValueError('Wrap parameter-only blocks; keep rotary buffers outside')
        if any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p
               for m in module.modules()):
            raise ValueError('Stochastic recomputation unsupported')
        entries = list(module.named_parameters())
        if not entries or any(p.device.type != 'cpu' or not p.requires_grad for _, p in entries):
            raise ValueError('Initialize from a CPU module')
        self.spec = [(n, p.shape, p.numel()) for n, p in entries]
        self.numel = sum(p.numel() for _, p in entries)
        self.world, self.rank = dist.get_world_size(), dist.get_rank()
        self.size = (self.numel + self.world - 1) // self.world
        self.compute_dtype = compute_dtype
        self.reduce_average = False
        key = tuple(id(p) for _, p in entries)
        if shared is not None and key in shared:
            self.shard = shared[key]
        else:
            # CPU-only full loading; copy only the local interval to CUDA.
            flat = torch.zeros(self.size * self.world, dtype=torch.float32)
            offset = 0
            for _, p in entries:
                flat[offset:offset + p.numel()].copy_(p.detach().reshape(-1))
                offset += p.numel()
            self.shard = torch.nn.Parameter(flat[self.rank*self.size:(self.rank+1)*self.size].cuda())
            if shared is not None:
                shared[key] = self.shard
        module.to('meta')
        object.__setattr__(self, '_template', module)

    def train(self, mode=True):
        super().train(mode)
        self._template.train(mode)
        return self

    def gather(self, shard):
        # Gather BF16 to halve communication and transient parameter storage.
        local = copy_scale(shard,dtype=self.compute_dtype)
        full = torch.empty(self.size*self.world, device=shard.device, dtype=self.compute_dtype)
        dist.all_gather_into_tensor(full, local)
        return full

    @staticmethod
    def unpack(structure, constants, inputs):
        iterator = iter(inputs)
        return tree_unflatten([next(iterator) if is_tensor else value
                               for is_tensor, value in constants], structure)

    def call(self, full, args, kwargs):
        state, offset = {}, 0
        for name, shape, count in self.spec:
            state[name] = full[offset:offset+count].view(shape)
            offset += count
        return functional_call(self._template, state, args, kwargs, strict=True)

    def forward(self, *args, **kwargs):
        if kwargs.get('use_cache') or kwargs.get('output_attentions'):
            raise ValueError('Cache and attention output unsupported in training')
        leaves, structure = tree_flatten((args, kwargs))
        constants = [(isinstance(v, torch.Tensor), None if isinstance(v, torch.Tensor) else v)
                     for v in leaves]
        inputs = [v for v in leaves if isinstance(v, torch.Tensor)]
        out = _Call.apply(self, structure, constants, self.shard, *inputs)
        return (out,) if self._template.__class__.__name__.endswith('DecoderLayer') else out


def shard_qwen(model):
    """Wrap CPU-loaded HF Qwen3; handles embedding/head tied shard ownership."""
    shared = {}
    if getattr(model.config,'attention_dropout',0):
        raise ValueError('Attention dropout requires RNG-preserving recomputation')
    body = model.model
    body.embed_tokens = BlockFSDP(body.embed_tokens, shared=shared)
    for i in range(len(body.layers)):
        body.layers[i] = BlockFSDP(body.layers[i], shared=shared)
    body.norm = BlockFSDP(body.norm, shared=shared)
    model.lm_head = BlockFSDP(model.lm_head, shared=shared)
    body.rotary_emb.cuda()
    model.config.use_cache = False
    return model
