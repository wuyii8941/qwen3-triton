"""Parity check between the Tiny-Qwen PyTorch baseline and HF Qwen3.

This is intentionally a small configuration so it runs quickly. The state
dict names are kept aligned with Qwen3, making the mapping an explicit test
instead of a hand-written collection of one-off copies.
"""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from model import TinyQwenConfig, TinyQwenForCausalLM


def make_configs() -> tuple[TinyQwenConfig, Qwen3Config]:
    tiny = TinyQwenConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    hf = Qwen3Config(
        vocab_size=tiny.vocab_size,
        hidden_size=tiny.hidden_size,
        intermediate_size=tiny.intermediate_size,
        num_hidden_layers=tiny.num_hidden_layers,
        num_attention_heads=tiny.num_attention_heads,
        num_key_value_heads=tiny.num_key_value_heads,
        head_dim=tiny.head_dim,
        max_position_embeddings=tiny.max_position_embeddings,
        rope_theta=tiny.rope_theta,
        rms_norm_eps=tiny.rms_norm_eps,
        attention_bias=False,
        tie_word_embeddings=True,
        use_sliding_window=False,
        _attn_implementation="eager",
    )
    return tiny, hf


def test_tiny_qwen_matches_huggingface_logits() -> None:
    torch.manual_seed(1234)
    tiny_config, hf_config = make_configs()
    hf_model = Qwen3ForCausalLM(hf_config).eval()
    tiny_model = TinyQwenForCausalLM(tiny_config).eval()

    # Qwen3 and Tiny-Qwen intentionally use matching state-dict names.
    missing, unexpected = tiny_model.load_state_dict(hf_model.state_dict(), strict=False)
    assert missing == []
    assert unexpected == []

    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 11))
    labels = input_ids.clone()
    with torch.inference_mode():
        hf_output = hf_model(input_ids=input_ids, labels=labels, use_cache=False)
        tiny_output = tiny_model(input_ids=input_ids, labels=labels)

    torch.testing.assert_close(tiny_output["logits"], hf_output.logits, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(tiny_output["loss"], hf_output.loss, rtol=2e-4, atol=2e-4)
