"""Dense embedding with explicit Triton gather and deterministic backward.

Baseline backward uses one writer per unique token and hidden tile. It scans
positions in fixed blocks, avoiding floating point atomic adds and ATen scatter.
Optimization can replace this algorithm without changing its public semantics.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _lookup(IDS, W, Y, TOTAL: tl.constexpr, H: tl.constexpr, V: tl.constexpr,
            BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < TOTAL
    token = tl.load(IDS + i // H, valid, 0)
    tl.device_assert((~valid) | ((token >= 0) & (token < V)), 'embedding ID out of range')
    value = tl.load(W + token * H + i % H, valid & (token >= 0) & (token < V), 0)
    tl.store(Y + i, value, valid)


@triton.jit
def _zero(X, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + i, 0., i < N)


@triton.jit
def _backward(IDS, DY, DW, T: tl.constexpr, H: tl.constexpr,
              ALL: tl.constexpr, D: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0)
    token = tl.load(IDS + row)
    positions = tl.arange(0, ALL)
    previous = tl.load(IDS + positions, positions < row, -1)
    first = tl.sum(((positions < row) & (previous == token)).to(tl.int32), 0) == 0
    if first:
        col = tl.program_id(1) * D + tl.arange(0, D)
        acc = tl.full((D,), 0., tl.float32)
        for block in range(tl.cdiv(T, R)):
            pos = block * R + tl.arange(0, R)
            ids = tl.load(IDS + pos, pos < T, -1)
            g = tl.load(DY + pos[:, None] * H + col[None, :],
                        (pos[:, None] < T) & (col[None, :] < H) & (ids[:, None] == token), 0)
            acc += tl.sum(g.to(tl.float32), 0)
        tl.store(DW + token * H + col, acc, col < H)


class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, weight):
        if not ids.is_cuda or not weight.is_cuda or ids.device != weight.device:
            raise ValueError('Same-device CUDA inputs required; no fallback')
        if ids.dtype not in (torch.int32, torch.int64) or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError('Integer IDs and floating point weight required')
        if not ids.is_contiguous() or not weight.is_contiguous() or weight.ndim != 2 or ids.numel() < 1:
            raise ValueError('Contiguous nonempty IDs and matrix weight required')
        v, h = weight.shape
        if min(v, h) < 1:raise ValueError('Nonempty vocabulary and hidden dimension required')
        y = torch.empty((*ids.shape, h), device=weight.device, dtype=weight.dtype)
        _lookup[(triton.cdiv(y.numel(), 256),)](ids, weight, y, y.numel(), h, v, 256, debug=True)
        ctx.save_for_backward(ids)
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        return y

    @staticmethod
    def backward(ctx, dy):
        ids, = ctx.saved_tensors
        v, h = ctx.weight_shape
        dy = dy.contiguous()
        dw = torch.empty((v, h), device=dy.device, dtype=ctx.weight_dtype)
        _zero[(triton.cdiv(v * h, 1024),)](dw, v * h, 1024)
        _backward[(ids.numel(), triton.cdiv(h, 128))](
            ids, dy, dw, ids.numel(), h, triton.next_power_of_2(ids.numel()), 128, 32,
            num_warps=4, enable_fp_fusion=False)
        return None, dw


def embedding(ids, weight):
    return _Embedding.apply(ids, weight)
