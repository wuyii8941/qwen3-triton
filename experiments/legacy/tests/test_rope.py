import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb

from model import TinyQwenConfig
from model.qwen_tiny import RotaryEmbedding
from ops.reference.rope import apply_rope_reference, rotate_half_reference
from ops.triton.rope import apply_rope_triton


def test_split_half_rotation_manual() -> None:
    x = torch.arange(8).float()
    expected = torch.tensor([-4.0, -5.0, -6.0, -7.0, 0.0, 1.0, 2.0, 3.0])
    torch.testing.assert_close(rotate_half_reference(x), expected)


def test_rope_reference_matches_hf() -> None:
    config = Qwen3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        rope_theta=1_000_000.0,
    )
    hf_rope = Qwen3RotaryEmbedding(config)
    hidden = torch.randn(2, 5, 64)
    position_ids = torch.tensor([[0, 1, 2, 7, 8], [3, 4, 5, 6, 9]])
    cos, sin = hf_rope(hidden, position_ids)
    q = torch.randn(2, 4, 5, 16)
    k = torch.randn(2, 2, 5, 16)
    expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
    actual_q, actual_k = apply_rope_reference(q, k, cos, sin)
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RoPE Triton tests require CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_triton_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    batch, q_heads, kv_heads, seq_len, head_dim = 2, 4, 2, 7, 16
    q = torch.randn(batch, q_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    cos = torch.randn(batch, seq_len, head_dim, device="cuda", dtype=dtype)
    sin = torch.randn(batch, seq_len, head_dim, device="cuda", dtype=dtype)
    expected_q, expected_k = apply_rope_reference(q, k, cos, sin)
    actual_q, actual_k = apply_rope_triton(q, k, cos, sin)
    torch.testing.assert_close(actual_q, expected_q, rtol=3e-2, atol=3.2e-2)
    torch.testing.assert_close(actual_k, expected_k, rtol=3e-2, atol=3.2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RoPE Triton tests require CUDA")
def test_rope_triton_backward_matches_reference() -> None:
    torch.manual_seed(1)
    q_ref = torch.randn(2, 4, 5, 16, device="cuda", requires_grad=True)
    k_ref = torch.randn(2, 2, 5, 16, device="cuda", requires_grad=True)
    q_custom = q_ref.detach().clone().requires_grad_()
    k_custom = k_ref.detach().clone().requires_grad_()
    cos = torch.randn(2, 5, 16, device="cuda")
    sin = torch.randn(2, 5, 16, device="cuda")
    grad_q = torch.randn_like(q_ref)
    grad_k = torch.randn_like(k_ref)
    ref_q, ref_k = apply_rope_reference(q_ref, k_ref, cos, sin)
    ref_q.backward(grad_q)
    ref_k.backward(grad_k)
    custom_q, custom_k = apply_rope_triton(q_custom, k_custom, cos, sin)
    torch.autograd.backward((custom_q, custom_k), (grad_q, grad_k))
    torch.testing.assert_close(q_custom.grad, q_ref.grad)
    torch.testing.assert_close(k_custom.grad, k_ref.grad)
