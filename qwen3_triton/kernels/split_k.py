"""Split-K matrix multiplication with FP32 partials and a fixed reduction."""
import torch
import triton
import triton.language as tl


@triton.jit
def _split_gemm(A, B, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr,
                BN: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr,
                TK: tl.constexpr, GROUP: tl.constexpr, SPLITS: tl.constexpr):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    nm, nn = tl.cdiv(M, TM), tl.cdiv(N, TN)
    width = GROUP * nn
    first = (pid // width) * GROUP
    height = tl.minimum(nm - first, GROUP)
    pm = first + (pid % width) % height
    pn = (pid % width) // height
    rows = pm * TM + tl.arange(0, TM)
    cols = pn * TN + tl.arange(0, TN)
    offsets = tl.arange(0, TK)
    iterations: tl.constexpr = tl.cdiv(K, TK * SPLITS)
    acc = tl.zeros((TM, TN), tl.float32)
    for block in range(iterations):
        ks = (split * iterations + block) * TK + offsets
        a = tl.load(A + rows[:, None] * AM + ks[None, :] * AK,
                    (rows[:, None] < M) & (ks[None, :] < K), 0)
        b = tl.load(B + ks[:, None] * BK + cols[None, :] * BN,
                    (ks[:, None] < K) & (cols[None, :] < N), 0)
        acc += tl.dot(a, b, input_precision='ieee')
    tl.store(P + split * M * N + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _reduce_split(P, C, SIZE: tl.constexpr, SPLITS: tl.constexpr,
                  BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, triton.next_power_of_2(SPLITS))
    values = tl.load(P + splits[:, None] * SIZE + offsets[None, :],
                     (splits[:, None] < SPLITS) & (offsets[None, :] < SIZE), 0)
    result = tl.sum(values, axis=0)
    tl.store(C + offsets, result, offsets < SIZE)


def split_matmul_into(a, b, out, partials, config, splits):
    """Write partials and reduce them into the provided output."""
    tm, tn, tk, group, warps, stages = config
    m, k = a.shape
    n = b.shape[1]
    assert a.dtype == b.dtype == out.dtype == torch.bfloat16
    assert out.shape == (m, n) and partials.shape == (splits, m, n)
    assert partials.dtype == torch.float32
    kernel = _split_gemm[(triton.cdiv(m, tm) * triton.cdiv(n, tn), splits)](
        a, b, partials, m, n, k, *a.stride(), *b.stride(),
        tm, tn, tk, group, splits, num_warps=warps, num_stages=stages)
    _reduce_split[(triton.cdiv(m * n, 512),)](partials, out, m * n, splits, 512)
    return kernel
