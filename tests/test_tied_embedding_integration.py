"""Tiny real HF Qwen integration: accumulation, recompute and state format."""
import copy
import pytest
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import Qwen3Config, Qwen3ForCausalLM
from benchmarks.check_tied_embedding import patch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('checkpoint,compiled', [(False,False),(True,False),(False,True)])
@pytest.mark.parametrize('consume_proxy',[False,True])
def test_accumulation_checkpoint_state(dtype, checkpoint, compiled, consume_proxy):
    torch.manual_seed(81)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, tie_word_embeddings=True, attention_dropout=0.0)
    cfg._attn_implementation = 'sdpa'
    reference = Qwen3ForCausalLM(cfg).to(device='cuda',dtype=dtype).train()
    candidate = copy.deepcopy(reference)
    keys = set(candidate.state_dict())
    undo = patch(candidate,consume_proxy=consume_proxy)
    for model in (reference,candidate):
        if checkpoint:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    def get_fn(model):
        def fn(ids): return model(ids,labels=ids,use_cache=False).loss
        return torch.compile(fn,fullgraph=True,dynamic=False) if compiled else fn
    fns = [get_fn(model) for model in (reference,candidate)]
    # Two microbatches, different IDs; shared gradient accumulation must work.
    with sdpa_kernel(SDPBackend.MATH):
        for _ in range(2):
            ids = torch.randint(0,97,(2,15),device='cuda')
            losses = []
            for fn in fns:
                loss = fn(ids)/2
                losses.append(float(loss.detach()))
                loss.backward()
            assert abs(losses[0]-losses[1]) < (1e-6 if dtype==torch.float32 else .01)
    for (name,a),(other,b) in zip(reference.named_parameters(),candidate.named_parameters()):
        assert name == other and a.grad is not None and b.grad is not None
        rel = (a.grad.float()-b.grad.float()).norm()/a.grad.float().norm().clamp_min(1e-12)
        assert float(rel) < (2e-5 if dtype==torch.float32 else .02), (name,float(rel))
    assert set(candidate.state_dict()) == keys
    fresh = Qwen3ForCausalLM(cfg).to(device='cuda',dtype=dtype)
    fresh.load_state_dict(candidate.state_dict())
    assert fresh.model.embed_tokens.weight is fresh.lm_head.weight
    undo()
    assert candidate.model.embed_tokens.weight is candidate.lm_head.weight
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        torch.testing.assert_close(reference(ids,use_cache=False).logits,candidate(ids,use_cache=False).logits,rtol=0,atol=0)
