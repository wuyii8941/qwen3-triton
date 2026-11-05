"""Triton RMSNorm forward and backward kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    x_float = x.to(tl.float32)
    mean_square = tl.sum(x_float * x_float, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_square + eps)
    y = x_float * inv_rms * w.to(tl.float32)
    tl.store(y_ptr + row * n_cols + offsets, y, mask=mask)


@triton.jit
def _rms_norm_backward_kernel(
    x_ptr,
    w_ptr,
    grad_y_ptr,
    grad_x_ptr,
    grad_w_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_y = tl.load(grad_y_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)

    mean_square = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_square + eps)
    weighted_grad = grad_y * w
    correction = tl.sum(weighted_grad * x, axis=0) / n_cols
    grad_x = (weighted_grad - x * inv_rms * inv_rms * correction) * inv_rms
    tl.store(grad_x_ptr + row * n_cols + offsets, grad_x, mask=mask)

    grad_w = grad_y * x * inv_rms
    tl.atomic_add(grad_w_ptr + offsets, grad_w, mask=mask)


def _forward_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("rms_norm_triton requires CUDA tensors")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported input dtype: {x.dtype}")
    if weight.dtype != x.dtype:
        raise TypeError("x and weight must have the same dtype")
    if x.shape[-1] != weight.numel():
        raise ValueError("weight length must equal x.shape[-1]")

    x_2d = x.contiguous().view(-1, x.shape[-1])
    weight_1d = weight.contiguous().view(-1)
    output = torch.empty_like(x_2d)
    n_cols = x_2d.shape[1]
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError("hidden dimension is too large for this first kernel")
    _rms_norm_forward_kernel[(x_2d.shape[0],)](
        x_2d,
        weight_1d,
        output,
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    return output.view_as(x)


def _backward_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_2d = x.contiguous().view(-1, x.shape[-1])
    grad_output_2d = grad_output.contiguous().view_as(x_2d)
    weight_1d = weight.contiguous().view(-1)
    grad_x = torch.empty_like(x_2d)
    grad_weight_float = torch.zeros(weight_1d.shape, device=weight.device, dtype=torch.float32)
    n_cols = x_2d.shape[1]
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError("hidden dimension is too large for this first kernel")
    _rms_norm_backward_kernel[(x_2d.shape[0],)](
        x_2d,
        weight_1d,
        grad_output_2d,
        grad_x,
        grad_weight_float,
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    return grad_x.view_as(x), grad_weight_float.to(weight.dtype)


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        output = _forward_triton(x, weight, eps)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        # The backward kernel accumulates grad_weight in fp32 and casts it to
        # the parameter dtype only after all rows have contributed.
        grad_x, grad_weight = _backward_triton(x, weight, grad_output, eps)
        return grad_x, grad_weight, None


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply Qwen RMSNorm with a Triton forward kernel."""
    return _RMSNormFunction.apply(x, weight, eps)
