"""Triton implementation of the Qwen SwiGLU activation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _silu_and_mul_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute ``(gate * sigmoid(gate)) * up`` for one flattened tile."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    # Implement SiLU directly in FP32 and cast on store.
    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)
    # One exponential is sufficient: exp(-x)=inf for very negative x gives
    # sigmoid=0, while exp(-x)=0 for very positive x gives sigmoid=1.
    sigmoid = 1.0 / (1.0 + tl.exp(-gate_f))
    result = gate_f * sigmoid * up_f
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _silu_and_mul_backward_kernel(
    gate_ptr,
    up_ptr,
    grad_out_ptr,
    grad_gate_ptr,
    grad_up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute both input gradients for ``silu(gate) * up``."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Use the same sigmoid expression as forward so forward/backward agree.
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    grad_up = grad_out * gate * sigmoid
    grad_gate = grad_out * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
    tl.store(grad_gate_ptr + offsets, grad_gate, mask=mask)
    tl.store(grad_up_ptr + offsets, grad_up, mask=mask)


class _TritonSiluAndMul(torch.autograd.Function):
    """Triton forward and backward for the SwiGLU activation."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        n_elements = gate.numel()
        output = torch.empty_like(gate)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _silu_and_mul_kernel[grid](
            gate, up, output, n_elements
        )
        ctx.save_for_backward(gate, up)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        n_elements = gate.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _silu_and_mul_backward_kernel[grid](
            gate,
            up,
            grad_output.contiguous(),
            grad_gate,
            grad_up,
            n_elements,
        )
        return grad_gate, grad_up


def silu_and_mul_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute Qwen SwiGLU activation with a Triton forward kernel."""
    if not (gate.is_cuda and up.is_cuda):
        raise ValueError("silu_and_mul_triton requires CUDA tensors")
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    if not (gate.is_floating_point() and up.is_floating_point()):
        raise TypeError("gate and up must be floating-point tensors")
    if gate.dtype != up.dtype:
        raise TypeError("gate and up must have the same dtype")
    return _TritonSiluAndMul.apply(gate.contiguous(), up.contiguous())
