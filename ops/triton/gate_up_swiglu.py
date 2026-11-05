"""Triton fused gate/up GEMM followed by SwiGLU activation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .silu_and_mul import _silu_and_mul_backward_kernel


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["m", "n", "k"],
)
@triton.jit
def _gate_up_swiglu_kernel(
    x_ptr,
    gate_w_ptr,
    up_w_ptr,
    out_ptr,
    m,
    n,
    k,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_K: tl.constexpr,
    GROUP_M: tl.constexpr = 0,
    HF_ROUND: tl.constexpr = False,
):
    """Compute two GEMM tiles and apply SwiGLU before the only global store."""
    if GROUP_M == 0:
        # Original 2-D ordering, retained for existing training/autograd users.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    else:
        # Experimental grouped-M order: neighboring programs reuse weight tiles
        # across several M tiles before moving to the next N tile.
        pid = tl.program_id(0)
        num_m = tl.cdiv(m, BLOCK_M)
        num_n = tl.cdiv(n, BLOCK_N)
        group = pid // (GROUP_M * num_n)
        first_m = group * GROUP_M
        group_size = tl.minimum(num_m - first_m, GROUP_M)
        within = pid % (GROUP_M * num_n)
        pid_m = first_m + within % group_size
        pid_n = within // group_size
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    row_mask = rows < m
    col_mask = cols < n

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, tl.cdiv(MAX_K, BLOCK_K)):
        k_offsets = start_k * BLOCK_K + ks
        k_mask = k_offsets < k
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + k_offsets[None, :] * stride_xk,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        gate_w = tl.load(
            gate_w_ptr + cols[:, None] * stride_wn + k_offsets[None, :] * stride_wk,
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        up_w = tl.load(
            up_w_ptr + cols[:, None] * stride_wn + k_offsets[None, :] * stride_wk,
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc_gate += tl.dot(x, tl.trans(gate_w), out_dtype=tl.float32, input_precision="ieee")
        acc_up += tl.dot(x, tl.trans(up_w), out_dtype=tl.float32, input_precision="ieee")

    if HF_ROUND:
        acc_gate = acc_gate.to(out_ptr.dtype.element_ty).to(tl.float32)
        acc_up = acc_up.to(out_ptr.dtype.element_ty).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-acc_gate))
    activated = acc_gate * sigmoid
    if HF_ROUND:
        activated = activated.to(out_ptr.dtype.element_ty).to(tl.float32)
    result = activated * acc_up
    out = rows[:, None] * stride_om + cols[None, :] * stride_on
    tl.store(out_ptr + out, result, mask=row_mask[:, None] & col_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["m", "n", "k"],
)
@triton.jit
def _dual_linear_kernel(
    x_ptr, gate_w_ptr, up_w_ptr, gate_ptr, up_ptr,
    m, n, k,
    stride_xm, stride_xk, stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    MAX_K: tl.constexpr,
):
    """Recompute both linear projections for the backward pass."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(0, tl.cdiv(MAX_K, BLOCK_K)):
        kk = start_k * BLOCK_K + tl.arange(0, BLOCK_K)
        km = kk < k
        x = tl.load(x_ptr + rows[:, None] * stride_xm + kk[None, :] * stride_xk,
                    mask=row_mask[:, None] & km[None, :], other=0.0)
        gw = tl.load(gate_w_ptr + cols[:, None] * stride_wn + kk[None, :] * stride_wk,
                     mask=col_mask[:, None] & km[None, :], other=0.0)
        uw = tl.load(up_w_ptr + cols[:, None] * stride_wn + kk[None, :] * stride_wk,
                     mask=col_mask[:, None] & km[None, :], other=0.0)
        acc_gate += tl.dot(x, tl.trans(gw), out_dtype=tl.float32, input_precision="ieee")
        acc_up += tl.dot(x, tl.trans(uw), out_dtype=tl.float32, input_precision="ieee")
    out = rows[:, None] * n + cols[None, :]
    tl.store(gate_ptr + out, acc_gate, mask=row_mask[:, None] & col_mask[None, :])
    tl.store(up_ptr + out, acc_up, mask=row_mask[:, None] & col_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["m", "n", "k"],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr, m, n, k,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    MAX_K: tl.constexpr,
):
    """Generic FP32-accumulating tiled matmul used by the backward path."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(0, tl.cdiv(MAX_K, BLOCK_K)):
        kk = start_k * BLOCK_K + tl.arange(0, BLOCK_K)
        km = kk < k
        a = tl.load(a_ptr + rows[:, None] * stride_am + kk[None, :] * stride_ak,
                    mask=row_mask[:, None] & km[None, :], other=0.0)
        b = tl.load(b_ptr + kk[:, None] * stride_bk + cols[None, :] * stride_bn,
                    mask=km[:, None] & col_mask[None, :], other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32, input_precision="ieee")
    out = rows[:, None] * stride_cm + cols[None, :] * stride_cn
    tl.store(c_ptr + out, acc, mask=row_mask[:, None] & col_mask[None, :])


class _GateUpSwiGLU(torch.autograd.Function):
    """Fused Triton forward with a PyTorch reference backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor):
        original_shape = x.shape
        m, k = x.reshape(-1, x.shape[-1]).shape
        n = gate_weight.shape[0]
        x_2d = x.reshape(m, k).contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()
        output = torch.empty((m, n), device=x.device, dtype=x.dtype)
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(n, meta["BLOCK_N"]),
        )
        _gate_up_swiglu_kernel[grid](
            x_2d,
            gate_weight,
            up_weight,
            output,
            m,
            n,
            k,
            x_2d.stride(0),
            x_2d.stride(1),
            gate_weight.stride(0),
            gate_weight.stride(1),
            output.stride(0),
            output.stride(1),
            MAX_K=k,
        )
        ctx.save_for_backward(x_2d, gate_weight, up_weight)
        ctx.original_shape = original_shape
        return output.reshape(*original_shape[:-1], n)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, gate_weight, up_weight = ctx.saved_tensors
        x = x.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()
        m, k = x.shape
        n = gate_weight.shape[0]

        # Recompute the two projections with Triton rather than invoking two
        # PyTorch Linear operations in backward.
        gate = torch.empty((m, n), device=x.device, dtype=x.dtype)
        up = torch.empty_like(gate)
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
        _dual_linear_kernel[grid](
            x, gate_weight, up_weight, gate, up, m, n, k,
            x.stride(0), x.stride(1), gate_weight.stride(0), gate_weight.stride(1),
            MAX_K=k,
        )

        grad = grad_output.reshape(m, n).contiguous()
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        _silu_and_mul_backward_kernel[lambda meta: (triton.cdiv(m * n, meta["BLOCK_SIZE"]),)](
            gate.reshape(-1), up.reshape(-1), grad.reshape(-1),
            grad_gate.reshape(-1), grad_up.reshape(-1), m * n,
        )

        def matmul(a, b, out, am, an, ak):
            mm_grid = lambda meta: (triton.cdiv(am, meta["BLOCK_M"]), triton.cdiv(an, meta["BLOCK_N"]))
            _matmul_kernel[mm_grid](
                a, b, out, am, an, ak,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                out.stride(0), out.stride(1), MAX_K=ak,
            )

        # dX = d_gate @ W_gate + d_up @ W_up.
        grad_x = torch.zeros((m, k), device=x.device, dtype=torch.float32)
        matmul(grad_gate, gate_weight, grad_x, m, k, n)
        grad_x_up = torch.empty_like(grad_x)
        matmul(grad_up, up_weight, grad_x_up, m, k, n)
        grad_x = (grad_x + grad_x_up).to(x.dtype)

        # dW = d_projection.T @ X.  Strides view d_gate/d_up as transposed.
        grad_gate_weight = torch.empty((n, k), device=x.device, dtype=torch.float32)
        grad_up_weight = torch.empty_like(grad_gate_weight)
        mm_grid = lambda meta: (triton.cdiv(n, meta["BLOCK_M"]), triton.cdiv(k, meta["BLOCK_N"]))
        _matmul_kernel[mm_grid](
            grad_gate, x, grad_gate_weight, n, k, m,
            grad_gate.stride(1), grad_gate.stride(0), x.stride(0), x.stride(1),
            grad_gate_weight.stride(0), grad_gate_weight.stride(1), MAX_K=m,
        )
        _matmul_kernel[mm_grid](
            grad_up, x, grad_up_weight, n, k, m,
            grad_up.stride(1), grad_up.stride(0), x.stride(0), x.stride(1),
            grad_up_weight.stride(0), grad_up_weight.stride(1), MAX_K=m,
        )
        return grad_x.reshape(ctx.original_shape), grad_gate_weight.to(gate_weight.dtype), grad_up_weight.to(up_weight.dtype)


def gate_up_swiglu_triton(
    x: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor
) -> torch.Tensor:
    """Compute ``silu(linear(x, gate_weight)) * linear(x, up_weight)``."""
    if not all(t.is_cuda for t in (x, gate_weight, up_weight)):
        raise ValueError("gate_up_swiglu_triton requires CUDA tensors")
    if x.ndim < 2 or gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("expected x [..., K] and matching weights [N, K]")
    if x.shape[-1] != gate_weight.shape[1]:
        raise ValueError("x and weight K dimensions must match")
    if not (x.is_floating_point() and gate_weight.is_floating_point() and up_weight.is_floating_point()):
        raise TypeError("all inputs must be floating-point tensors")
    if not (x.dtype == gate_weight.dtype == up_weight.dtype):
        raise TypeError("all inputs must have the same dtype")
    return _GateUpSwiGLU.apply(x, gate_weight, up_weight)
