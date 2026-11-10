"""Local copy/scale primitive used by the custom block-sharding backend."""
import torch
import triton
import triton.language as tl


@triton.jit
def _copy_scale(src, dst, size, scale, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + i, i < size, other=0).to(tl.float32)
    tl.store(dst + i, value * scale, i < size)


def copy_scale(src, scale=1., *, dtype=None):
    out = torch.empty_like(src, dtype=dtype or src.dtype)
    _copy_scale[(triton.cdiv(src.numel(), 256),)](src, out, src.numel(), scale, BLOCK=256)
    return out
