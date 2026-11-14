"""Match PyTorch 2.10 CUDA's vectorized row-mean summation order.

A generic tree sum can shift BF16 normalization at rounding boundaries.
Reduce.cuh uses four independent accumulators per lane, combines them in
order, then reduces warp offsets 16,8,4,2,1. Preserve that order explicitly.
The supported shapes use one warp per output (no cross-warp/CTA reduction).
"""
import torch
import triton
import triton.language as tl

_SUPPORTED_TORCH = torch.__version__.split('+')[0] == '2.10.0' and torch.version.hip is None


def supports_hf_reduction(x):
    cols = x.shape[-1]
    rows = x.numel() // cols if cols else 0
    return (_SUPPORTED_TORCH and x.is_cuda and x.is_contiguous() and rows >= 16
            and 128 <= cols <= 4096 and cols & (cols - 1) == 0)


@triton.jit
def hf_mean_square(x_ptr, base, COLS: tl.constexpr):
    lane = tl.arange(0, 32)
    a = tl.full((32,), 0., tl.float32)
    b = tl.full((32,), 0., tl.float32)
    c = tl.full((32,), 0., tl.float32)
    d = tl.full((32,), 0., tl.float32)
    for start in range(COLS // 128):
        offset = base + start * 128 + lane * 4
        x0 = tl.load(x_ptr + offset).to(tl.float32)
        x1 = tl.load(x_ptr + offset + 1).to(tl.float32)
        x2 = tl.load(x_ptr + offset + 2).to(tl.float32)
        x3 = tl.load(x_ptr + offset + 3).to(tl.float32)
        a = a + x0 * x0
        b = b + x1 * x1
        c = c + x2 * x2
        d = d + x3 * x3
    value = ((a + b) + c) + d
    for step in tl.static_range(5):
        value = value + tl.gather(value, lane ^ (16 >> step), axis=0)
    return tl.sum(tl.where(lane == 0, value, 0.), axis=0) * (1. / COLS)
