from __future__ import annotations

import torch


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen RMSNorm reference: normalize only the final dimension."""
    input_dtype = x.dtype
    x_float = x.float()
    weight_float = weight.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    normalized = x_float * torch.rsqrt(variance + eps)
    return (normalized * weight_float).to(input_dtype)
