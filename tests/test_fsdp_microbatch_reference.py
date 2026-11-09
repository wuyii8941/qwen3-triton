import copy
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from benchmarks.fsdp_reference import native_microbatch_reference


@pytest.mark.parametrize('tied',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_fresh_microbatch_casts_match_independent_master_gradient_accumulation(tied,dtype):
    torch.manual_seed(65)
    cfg=Qwen3Config(vocab_size=37,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        tie_word_embeddings=tied,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(cfg).train();other=copy.deepcopy(model)
    ids=torch.randint(37,(2,11))
    output=native_microbatch_reference(model,ids,dtype=dtype,tied=tied,accumulation=2)
    output.loss.backward()
    expected=[]
    for micro in ids.chunk(2):
        part=native_microbatch_reference(other,micro,dtype=dtype,tied=tied)
        (part.loss/2).backward();expected.append(part.logits.detach())
    torch.testing.assert_close(output.logits,torch.cat(expected),atol=0,rtol=0)
    for name,p in model.named_parameters():
        torch.testing.assert_close(p.grad,other.get_parameter(name).grad,atol=1e-7,rtol=1e-6)
    if dtype==torch.float32:
        other.zero_grad(set_to_none=True)
        full=native_microbatch_reference(other,ids,dtype=dtype,tied=tied)
        full.loss.backward()
        torch.testing.assert_close(output.loss,full.loss,atol=1e-6,rtol=1e-6)
        for name,p in model.named_parameters():
            torch.testing.assert_close(p.grad,other.get_parameter(name).grad,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('count',[0,-1,True,3])
def test_reject_invalid_partition(count):
    with pytest.raises(ValueError):
        native_microbatch_reference(None,torch.ones(2,3),dtype=torch.float32,tied=False,accumulation=count)
