import torch
import pytest

from model import TinyQwenConfig, TinyQwenForCausalLM
from model.qwen_tiny import QwenAttention, QwenRMSNorm, RotaryEmbedding


def small_config() -> TinyQwenConfig:
    return TinyQwenConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def test_rms_norm_matches_definition() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 64)
    module = QwenRMSNorm(64, 1e-6)
    actual = module(x)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    expected = x * torch.rsqrt(variance + 1e-6)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rotary_embedding_shape() -> None:
    config = small_config()
    hidden = torch.zeros(2, 7, config.hidden_size)
    position_ids = torch.arange(7).unsqueeze(0).expand(2, -1)
    cos, sin = RotaryEmbedding(config)(hidden, position_ids)
    assert cos.shape == (2, 7, config.head_dim)
    assert sin.shape == cos.shape
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()


def test_rope_uses_split_half_rotation() -> None:
    from model.qwen_tiny import _rotate_half

    x = torch.arange(8).float()
    torch.testing.assert_close(_rotate_half(x), torch.tensor([-4.0, -5.0, -6.0, -7.0, 0.0, 1.0, 2.0, 3.0]))


def test_gqa_attention_shape_and_gradients() -> None:
    torch.manual_seed(0)
    config = small_config()
    module = QwenAttention(config)
    hidden = torch.randn(2, 7, config.hidden_size, requires_grad=True)
    position_ids = torch.arange(7).unsqueeze(0).expand(2, -1)
    position_embeddings = RotaryEmbedding(config)(hidden, position_ids)
    output = module(hidden, position_embeddings)
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_full_model_forward_loss_and_backward() -> None:
    torch.manual_seed(0)
    config = small_config()
    model = TinyQwenForCausalLM(config)
    input_ids = torch.randint(config.vocab_size, (2, 8))
    result = model(input_ids, labels=input_ids)
    assert result["logits"].shape == (2, 8, config.vocab_size)
    assert result["loss"].ndim == 0
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_training_step_changes_parameters() -> None:
    torch.manual_seed(0)
    model = TinyQwenForCausalLM(small_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.arange(8).unsqueeze(0).repeat(2, 1)
    before = model.model.embed_tokens.weight.detach().clone()
    result = model(input_ids, labels=input_ids)
    result["loss"].backward()
    optimizer.step()
    after = model.model.embed_tokens.weight.detach()
    assert not torch.equal(before, after)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton integration requires CUDA")
def test_tiny_qwen_triton_rms_norm_matches_torch() -> None:
    torch.manual_seed(7)
    torch_config = small_config()
    triton_config = small_config()
    triton_config.rms_norm_impl = "triton"
    torch_model = TinyQwenForCausalLM(torch_config).cuda().eval()
    triton_model = TinyQwenForCausalLM(triton_config).cuda().eval()
    triton_model.load_state_dict(torch_model.state_dict())
    input_ids = torch.randint(torch_config.vocab_size, (2, 9), device="cuda")
    with torch.inference_mode():
        torch_output = torch_model(input_ids, labels=input_ids)
        triton_output = triton_model(input_ids, labels=input_ids)
    torch.testing.assert_close(triton_output["logits"], torch_output["logits"], rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(triton_output["loss"], torch_output["loss"], rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton integration requires CUDA")
def test_tiny_qwen_fused_qk_norm_rope_matches_torch() -> None:
    torch.manual_seed(8)
    torch_config = small_config()
    fused_config = small_config()
    fused_config.qk_norm_rope_impl = "fused_triton"
    torch_model = TinyQwenForCausalLM(torch_config).cuda()
    fused_model = TinyQwenForCausalLM(fused_config).cuda()
    fused_model.load_state_dict(torch_model.state_dict())
    input_ids = torch.randint(torch_config.vocab_size, (2, 9), device="cuda")

    torch_result = torch_model(input_ids, labels=input_ids)
    fused_result = fused_model(input_ids, labels=input_ids)
    torch.testing.assert_close(fused_result["logits"], torch_result["logits"], rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(fused_result["loss"], torch_result["loss"], rtol=3e-3, atol=3e-3)

    torch_result["loss"].backward()
    fused_result["loss"].backward()
    for name, torch_param in torch_model.named_parameters():
        fused_param = dict(fused_model.named_parameters())[name]
        if torch_param.grad is not None:
            torch.testing.assert_close(fused_param.grad, torch_param.grad, rtol=5e-3, atol=5e-3)
