"""End-to-end compact-GQA prefill/decode pipeline.

This module deliberately keeps the cache simple: one preallocated contiguous
K/V tensor per request.  It is a correctness and benchmarking baseline before
introducing paged cache or continuous batching.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .attention import attention_triton_decode, attention_triton_qwen_prefill
from .decode_split_kv import attention_decode_split


class ContiguousKVCache:
    """Static contiguous KV cache in compact ``[B, Hkv, S, D]`` form."""

    def __init__(self, batch: int, kv_heads: int, max_seq_len: int, head_dim: int,
                 *, device: torch.device, dtype: torch.dtype) -> None:
        if batch <= 0 or kv_heads <= 0 or max_seq_len <= 0 or head_dim <= 0:
            raise ValueError("cache dimensions must be positive")
        self.k = torch.empty((batch, kv_heads, max_seq_len, head_dim), device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.max_seq_len = max_seq_len
        self.seq_len = 0

    @property
    def batch(self) -> int:
        return self.k.shape[0]

    @property
    def kv_heads(self) -> int:
        return self.k.shape[1]

    @property
    def head_dim(self) -> int:
        return self.k.shape[-1]

    def reset(self) -> None:
        self.seq_len = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if k_new.ndim != 4 or v_new.shape != k_new.shape:
            raise ValueError("new K/V must have shape [B, Hkv, T, D]")
        if k_new.shape[:2] != self.k.shape[:2] or k_new.shape[-1] != self.head_dim:
            raise ValueError("new K/V shape is incompatible with cache")
        n = k_new.shape[2]
        if self.seq_len + n > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        self.k[:, :, self.seq_len:self.seq_len + n].copy_(k_new)
        self.v[:, :, self.seq_len:self.seq_len + n].copy_(v_new)
        self.seq_len += n

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.seq_len <= 0:
            raise ValueError("KV cache is empty")
        return self.k[:, :, :self.seq_len], self.v[:, :, :self.seq_len]


class CompactGQAAttentionPipeline:
    """Prefill and decode using compact GQA tensors and a static KV cache."""

    def __init__(self, q_heads: int, kv_heads: int, head_dim: int = 128, *,
                 decode_impl: str = 'legacy', decode_splits: int = 8,
                 decode_group_heads: bool = True) -> None:
        if q_heads % kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads")
        if head_dim != 128:
            raise ValueError("the current Qwen pipeline targets head_dim=128")
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        if decode_impl not in ('legacy','split_kv'):
            raise ValueError('decode_impl must be legacy or split_kv')
        if not isinstance(decode_splits,int) or not 1<=decode_splits<=64:
            raise ValueError('decode_splits must be 1..64')
        self.decode_impl=decode_impl
        self.decode_splits=decode_splits
        self.decode_group_heads=decode_group_heads

    def prefill(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cache: ContiguousKVCache
    ) -> torch.Tensor:
        """Process a prompt and initialize the cache."""
        if q.ndim != 4 or q.shape[1] != self.q_heads or q.shape[2] <= 0:
            raise ValueError("q must have shape [B, Hq, S, 128]")
        if k.shape != v.shape or k.shape[:2] != (q.shape[0], self.kv_heads):
            raise ValueError("k/v must have shape [B, Hkv, S, 128]")
        cache.reset()
        cache.append(k, v)
        return attention_triton_qwen_prefill(q, k, v, scale=self.scale)

    def decode(
        self, q_new: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
        cache: ContiguousKVCache,
    ) -> torch.Tensor:
        """Append one token and attend over the full contiguous cache."""
        if q_new.shape != (cache.batch, self.q_heads, 1, self.head_dim):
            raise ValueError("q_new must have shape [B, Hq, 1, 128]")
        cache.append(k_new, v_new)
        k_all, v_all = cache.view()
        if self.decode_impl == 'split_kv':
            return attention_decode_split(q_new,k_all,v_all,scale=self.scale,
                splits=self.decode_splits,group_heads=self.decode_group_heads)
        return attention_triton_decode(q_new, k_all, v_all, scale=self.scale)


def sdpa_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Native PyTorch Flash Attention baseline; enable_gqa expands internally."""
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


def sdpa_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    """Native PyTorch Flash Attention decode baseline."""
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k_cache, v_cache, is_causal=False, enable_gqa=True)
