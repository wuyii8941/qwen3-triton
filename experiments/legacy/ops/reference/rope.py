from __future__ import annotations

import torch


def rotate_half_reference(x: torch.Tensor) -> torch.Tensor:
    """Qwen3 convention: pair the first and second halves of head_dim."""
    first, second = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3 RoPE to q/k.

    q/k have shape [batch, heads, seq, head_dim]; cos/sin have
    shape [batch, seq, head_dim].
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        q * cos + rotate_half_reference(q) * sin,
        k * cos + rotate_half_reference(k) * sin,
    )


def inverse_rotate_half_reference(x: torch.Tensor) -> torch.Tensor:
    """Apply the transpose (inverse) of Qwen3's split-half rotation."""
    first, second = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((second, -first), dim=-1)


def apply_rope_backward_reference(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference backward for :func:`apply_rope_reference`.

    ``grad_q`` and ``grad_k`` are gradients with respect to the rotated
    outputs and have shape ``[batch, heads, seq, head_dim]``.  The sine factor
    must be applied before the transpose of the rotation matrix, because its
    value is indexed by the output coordinate.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        grad_q * cos + inverse_rotate_half_reference(grad_q * sin),
        grad_k * cos + inverse_rotate_half_reference(grad_k * sin),
    )
