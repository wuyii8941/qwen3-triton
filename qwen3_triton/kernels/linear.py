"""Bias-free trainable Linear with Triton forward, dX and dW GEMMs.

Tensor allocation and stride-only transpose views use PyTorch. No ATen matmul,
cuBLAS, torch.compile or FlashAttention implementation is called here.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
          BM: tl.constexpr, BN_TILE: tl.constexpr, BK_TILE: tl.constexpr,
          GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN_TILE)
    if GROUP == 1:
        pm, pn = pid // nn, pid % nn
    else:
        width = GROUP * nn
        group = pid // width
        first = group * GROUP
        height = tl.minimum(nm - first, GROUP)
        pm = first + (pid % width) % height
        pn = (pid % width) // height
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN_TILE + tl.arange(0, BN_TILE)
    kk = tl.arange(0, BK_TILE)
    acc = tl.zeros((BM, BN_TILE), tl.float32)
    for block in range(tl.cdiv(K, BK_TILE)):
        ks = block * BK_TILE + kk
        a = tl.load(A + rows[:, None] * AM + ks[None, :] * AK,
                    (rows[:, None] < M) & (ks[None, :] < K), 0)
        b = tl.load(B + ks[:, None] * BK + cols[None, :] * BN,
                    (ks[:, None] < K) & (cols[None, :] < N), 0)
        acc += tl.dot(a, b, input_precision='ieee')
    tl.store(C + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def matmul(a, b, *, optimized=False, output_dtype=None):
    """A[M,K] @ B[K,N]; strided matrices, contiguous result, FP32 accumulation."""
    if not (a.is_cuda and b.is_cuda) or a.device != b.device:
        raise ValueError('Same-device CUDA matrices required; no fallback')
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('Expected [M,K] and [K,N]')
    if a.dtype != b.dtype or a.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError('Matching BF16/FP16/FP32 matrices required')
    if min(*a.shape, *b.shape) < 1 or min(*a.stride(), *b.stride()) < 1:
        raise ValueError('Nonempty, positive-stride matrices required')
    dtype = output_dtype or a.dtype
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError('Floating point output required')
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=dtype)
    bm, bn, bk, group = (64, 64, 32, 8) if optimized else (32, 64, 32, 1)
    _gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a, b, out, m, n, k, *a.stride(), *b.stride(), bm, bn, bk, group,
        num_warps=4)
    return out


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, optimized):
        if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
            raise ValueError('Expected X[...,K] and weight[N,K]')
        if not x.is_contiguous():
            raise ValueError('Linear input must be contiguous; no hidden layout copy')
        flat = x.view(-1, x.shape[-1])
        y = matmul(flat, weight.t(), optimized=optimized)
        ctx.save_for_backward(flat, weight)
        ctx.shape = x.shape
        ctx.optimized = optimized
        return y.view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        # Contiguous is a layout operation; arithmetic remains in Triton GEMMs.
        dy = dy.contiguous().view(-1, w.shape[0])
        dx = matmul(dy, w, optimized=ctx.optimized).view(ctx.shape) if ctx.needs_input_grad[0] else None
        dw = matmul(dy.t(), x, optimized=ctx.optimized) if ctx.needs_input_grad[1] else None
        return dx, dw, None


def linear(x, weight, *, optimized=False):
    return _Linear.apply(x, weight, optimized)
