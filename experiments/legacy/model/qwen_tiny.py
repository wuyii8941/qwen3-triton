"""A trainable, deliberately small Qwen-shaped language model.

The implementation uses ordinary PyTorch operations as a correctness
baseline. It is not intended to replace the Hugging Face Qwen3 model yet.
Each TODO marks a Qwen-specific extension or an optimization seam for later.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ops.reference.rms_norm import rms_norm_reference
from ops.reference.swiglu import swiglu_reference
from ops.triton.fused_qk_norm_rope import fused_qk_norm_rope_triton
from ops.triton.attention import attention_triton_autograd
from ops.triton.rms_norm import rms_norm_triton
from ops.triton.silu_and_mul import silu_and_mul_triton
from ops.triton.gate_up_swiglu import gate_up_swiglu_triton

from .config import TinyQwenConfig


class QwenRMSNorm(nn.Module):
    """Qwen-style RMSNorm baseline (no mean subtraction)."""

    def __init__(self, hidden_size: int, eps: float, impl: str = "torch") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        if impl not in {"torch", "triton"}:
            raise ValueError("RMSNorm impl must be 'torch' or 'triton'")
        self.impl = impl

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.impl == "triton":
            return rms_norm_triton(hidden_states, self.weight, self.eps)
        return rms_norm_reference(hidden_states, self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: TinyQwenConfig) -> None:
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = config.max_position_embeddings

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Qwen3 computes position embeddings once at model level and shares
        # them across decoder layers. position_ids is [batch, seq].
        positions = position_ids.to(device=hidden_states.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("bs,d->bsd", positions, self.inv_freq.to(hidden_states.device))
        # Qwen3 duplicates the frequency half ([freqs, freqs]); this pairs
        # with its split-half rotate_half implementation.
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(hidden_states.dtype)
        sin = emb.sin().to(hidden_states.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin are [batch, seq, head_dim]; q/k are [batch, heads, seq, head_dim].
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    # TODO(student): add rope_scaling once a checkpoint that uses it is in scope.
    return (q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin)


class QwenAttention(nn.Module):
    def __init__(self, config: TinyQwenConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_key_value_heads
        self.qk_norm_rope_impl = config.qk_norm_rope_impl
        self.attention_impl = config.attention_impl
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = QwenRMSNorm(self.head_dim, config.rms_norm_eps, config.rms_norm_impl)
        self.k_norm = QwenRMSNorm(self.head_dim, config.rms_norm_eps, config.rms_norm_impl)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        if self.qk_norm_rope_impl == "fused_triton":
            q, k = fused_qk_norm_rope_triton(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                cos,
                sin,
                self.q_norm.eps,
            )
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        else:
            q = self.q_norm(q).transpose(1, 2)
            k = self.k_norm(k).transpose(1, 2)
            q, k = apply_rotary(q, k, cos, sin)

        # Keep K/V in [B, Hkv, S, D].  Both SDPA (enable_gqa=True) and the
        # Triton kernels perform the query-head -> KV-head mapping internally;
        # do not materialize repeated K/V tensors here.
        if self.attention_impl == "triton":
            attended = attention_triton_autograd(q, k, v)
        else:
            attended = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attended)


class QwenMLP(nn.Module):
    def __init__(self, config: TinyQwenConfig) -> None:
        super().__init__()
        self.packed_gate_up = config.packed_gate_up
        self.fused_gate_up = config.fused_gate_up
        if self.packed_gate_up:
            # One GEMM produces [gate, up].  This is layout-compatible with
            # the two independent projections but changes the checkpoint
            # parameter names, so it is opt-in.
            self.gate_up_proj = nn.Linear(
                config.hidden_size, 2 * config.intermediate_size, bias=False
            )
            self.gate_proj = None
            self.up_proj = None
        else:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.gate_up_proj = None
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.swiglu_impl = config.swiglu_impl

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fused_gate_up:
            hidden = gate_up_swiglu_triton(
                hidden_states, self.gate_proj.weight, self.up_proj.weight
            )
            return self.down_proj(hidden)
        if self.packed_gate_up:
            gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        else:
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
        if self.swiglu_impl == "triton":
            hidden = silu_and_mul_triton(gate, up)
        else:
            hidden = swiglu_reference(gate, up)
        return self.down_proj(hidden)


class QwenDecoderBlock(nn.Module):
    def __init__(self, config: TinyQwenConfig) -> None:
        super().__init__()
        self.input_layernorm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps, config.rms_norm_impl)
        self.self_attn = QwenAttention(config)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps, config.rms_norm_impl)
        self.mlp = QwenMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), position_embeddings)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class TinyQwenModel(nn.Module):
    def __init__(self, config: TinyQwenConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config)
        self.layers = nn.ModuleList([QwenDecoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps, config.rms_norm_impl)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        return self.norm(hidden_states)


class TinyQwenForCausalLM(nn.Module):
    def __init__(self, config: TinyQwenConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyQwenConfig()
        self.model = TinyQwenModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # TODO(student): add the Hugging Face-compatible output/cache contract
        # and a verified state-dict mapping from Qwen3-1.7B.
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        output: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            output["loss"] = F.cross_entropy(shift_logits.float().view(-1, logits.size(-1)), shift_labels.view(-1))
        return output

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)
