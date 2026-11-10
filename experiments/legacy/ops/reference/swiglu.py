"""PyTorch reference for the Qwen SwiGLU activation."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute ``silu(gate) * up`` without changing the input dtype.

    ``gate`` and ``up`` may have any leading dimensions; the final dimension
    is the intermediate MLP dimension.  This function intentionally mirrors
    the eager PyTorch expression and serves as the numerical oracle for a
    future Triton fused kernel.
    """
    if gate.shape != up.shape:
        raise ValueError(f"gate and up must have the same shape, got {gate.shape} and {up.shape}")
    if not (gate.is_floating_point() and up.is_floating_point()):
        raise TypeError("SwiGLU inputs must be floating-point tensors")
    return F.silu(gate) * up
