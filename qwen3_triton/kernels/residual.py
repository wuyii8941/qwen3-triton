"""Residual addition; its derivative is an alias, not a numerical compute kernel."""
import torch
import triton
import triton.language as tl

@triton.jit
def _add(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i, i < N, 0).to(tl.float32)
    b = tl.load(B + i, i < N, 0).to(tl.float32)
    tl.store(C + i, a + b, i < N)

class _Residual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        if not a.is_cuda or a.device != b.device or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('Matching same-device CUDA tensors required; no fallback')
        if not a.is_contiguous() or not b.is_contiguous() or a.numel() < 1:
            raise ValueError('Contiguous nonempty residual inputs required')
        if a.dtype not in (torch.bfloat16, torch.float16, torch.float32):raise TypeError('Floating point inputs required')
        out = torch.empty_like(a)
        _add[(triton.cdiv(a.numel(), 512),)](a, b, out, a.numel(), 512)
        return out

    @staticmethod
    def backward(ctx, dy):
        return dy, dy

def residual_add(a, b):
    return _Residual.apply(a, b)
