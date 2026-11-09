import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from training.fsdp_loss import install_chunked_qwen_loss


def tiny(tied):
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, tie_word_embeddings=tied, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    return Qwen3ForCausalLM(config)


@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('chunk', [1, 7, 64])
@pytest.mark.parametrize('backward_impl', ['default','native_ce','native_ce_padded'])
def test_loss_gradients_updates_inference_and_module_hooks(tied, chunk, backward_impl):
    torch.manual_seed(83)
    reference = tiny(tied).train()
    model = copy.deepcopy(reference)
    weight = model.lm_head.weight
    install_chunked_qwen_loss(model, chunk_size=chunk, backward_impl=backward_impl)
    assert model.lm_head.weight is weight
    assert (model.model.embed_tokens.weight is weight) == tied
    assert model.state_dict().keys() == reference.state_dict().keys()
    calls = []
    handle = model.lm_head.register_forward_pre_hook(lambda *args: calls.append(1))
    a = torch.optim.AdamW(reference.parameters(), lr=1e-4)
    b = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):
        ids = torch.randint(97, (2, 11))
        labels = ids.clone(); labels[0, 3:6] = -100
        a.zero_grad(set_to_none=True); b.zero_grad(set_to_none=True)
        expected = reference(ids, labels=labels, use_cache=False)
        actual = model(ids, labels=labels, use_cache=False)
        assert actual.logits is None
        torch.testing.assert_close(actual.loss, expected.loss, atol=1e-6, rtol=1e-6)
        (expected.loss * .37).backward(); (actual.loss * .37).backward()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, reference.get_parameter(name).grad,
                                       atol=2e-7, rtol=3e-5)
        a.step(); b.step()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, reference.get_parameter(name), atol=2e-7, rtol=3e-5)
        with torch.no_grad():
            torch.testing.assert_close(model(ids, use_cache=False).logits,
                reference(ids, use_cache=False).logits, atol=2e-6, rtol=2e-5)
    assert len(calls) == 4
    handle.remove()


@pytest.mark.parametrize('kwargs', [dict(use_cache=True), dict(return_dict=False),
    dict(logits_to_keep=1), dict(num_items_in_batch=10), dict(past_key_values=object())])
def test_reject_unsupported_loss_semantics(kwargs):
    model = install_chunked_qwen_loss(tiny(False))
    ids = torch.ones(1, 5, dtype=torch.long)
    with pytest.raises(ValueError):
        model(ids, labels=ids, **kwargs)


def test_invalid_install_and_empty_label_gradients():
    model = tiny(True)
    for value in (0, -1, True):
        with pytest.raises(ValueError): install_chunked_qwen_loss(model, chunk_size=value)
    install_chunked_qwen_loss(model)
    with pytest.raises(ValueError): install_chunked_qwen_loss(model)
    ids = torch.ones(1, 5, dtype=torch.long)
    loss = model(ids, labels=torch.full_like(ids, -100)).loss
    assert torch.isnan(loss)
    loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


@pytest.mark.parametrize('tied', [False, True])
def test_bfloat16_compute_with_fp32_master_casts(tied):
    """CPU arithmetic gate, NOT a substitute for CUDA/FSDP hooks/collectives."""
    torch.manual_seed(101)
    reference = tiny(tied).train()
    model = install_chunked_qwen_loss(copy.deepcopy(reference), chunk_size=7)
    ids = torch.randint(97, (2, 11))
    # Match FSDP's distinct BF16 casts per module use of an FP32 master.
    a = {n: p.to(torch.bfloat16) for n, p in reference.named_parameters(remove_duplicate=tied)}
    b = {n: p.to(torch.bfloat16) for n, p in model.named_parameters(remove_duplicate=tied)}
    expected = torch.func.functional_call(reference, a, (ids,),
        dict(labels=ids, use_cache=False), tie_weights=tied)
    actual = torch.func.functional_call(model, b, (ids,),
        dict(labels=ids, use_cache=False), tie_weights=tied)
    torch.testing.assert_close(actual.loss, expected.loss, atol=2e-5, rtol=2e-5)
    (actual.loss * .37).backward(); (expected.loss * .37).backward()
    for name, p in model.named_parameters():
        q = reference.get_parameter(name)
        # Chunk-wise FP32 dW accumulation can differ from a single BF16 GEMM.
        relative = (p.grad - q.grad).norm() / q.grad.norm().clamp_min(1e-12)
        assert relative < .015, (name, float(relative))
