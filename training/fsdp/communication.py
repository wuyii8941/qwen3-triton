"""Opt-in BF16 transport with FP32 Torch reduction; native remains default.

Caller must guarantee original BF16 grads, no FP32 accumulation, no HSDP or
custom gradient divisor. The debug gate additionally checks exact roundtrip
on every rank before transport. All allocations honor FSDP's requested dtype.
"""
import torch
import torch.distributed as dist



class BF16ExchangeReduceScatter:
    def __init__(self, *, original_bf16, accumulation=1, backend='torch', debug=True):
        if original_bf16 is not True or accumulation != 1:
            raise ValueError('Original BF16 gradients and accumulation=1 required')
        if backend != 'torch':
            raise ValueError('Current training path uses Torch local reduction only')
        self.backend = backend
        self.debug = debug
        self.calls = 0

    def allocate(self, size, *, dtype, device):
        return torch.empty(size, dtype=dtype, device=device)

    @torch.no_grad()
    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        if async_op:
            raise ValueError('Async Work interface is not implemented')
        if op not in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
            raise ValueError('Only plain SUM/AVG supported; no custom divisor')
        world = dist.get_world_size(group)
        if (world < 2 or input_tensor.dtype != torch.float32 or
                output_tensor.dtype != torch.float32 or
                not input_tensor.is_contiguous() or not output_tensor.is_contiguous() or
                input_tensor.numel() != output_tensor.numel() * world or
                input_tensor.device != output_tensor.device or not output_tensor.numel()):
            raise ValueError('Contiguous FP32 input/output with matching equal shards required')
        packed = input_tensor.flatten().to(torch.bfloat16)
        if self.debug:
            # Collective decision prevents one rank raising while peers enter A2A.
            valid = torch.tensor(int(torch.isfinite(input_tensor).all().item() and
                                     torch.equal(packed.float(), input_tensor.flatten())),
                                 device=input_tensor.device, dtype=torch.int32)
            dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=group)
            if not valid.item():
                raise ValueError('Input is not an exact finite BF16 upcast on every rank')
        received = torch.empty_like(packed)
        dist.all_to_all_single(received, packed, group=group)
        cols = output_tensor.numel()
        scale = 1 / world if op == dist.ReduceOp.AVG else 1.
        values = received.view(world, cols).float()
        values.mul_(scale)
        torch.sum(values, dim=0, out=output_tensor.view(-1))
        self.calls += 1
        # NCCL and reduction enqueue on FSDP's current reduction stream.
        # Temporary tensors are allocated/used on that same stream; no hidden
        # side stream, input overwrite, output alias or persistent buffer reuse.
        return None
