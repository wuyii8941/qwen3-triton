import copy
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from training.ce_only_loss import install_ce_only_qwen_loss
from scripts.summarize_fsdp_memory import audit_result
from tests.test_fsdp_memory_summary import result


@pytest.mark.parametrize('tied',[False,True])
def test_adapter_keeps_head_parameters_keys_logits_and_backward(tied):
    torch.manual_seed(8)
    config=Qwen3Config(vocab_size=23,hidden_size=32,intermediate_size=48,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=tied,attention_dropout=0.)
    config._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(config);reference=copy.deepcopy(model)
    head=model.lm_head;parameters=list(model.parameters());keys=list(model.state_dict())
    install_ce_only_qwen_loss(model,chunk_size=3)
    assert head is model.lm_head and parameters==list(model.parameters())
    assert keys==list(model.state_dict())
    assert (model.lm_head.weight is model.model.embed_tokens.weight)==tied
    ids=torch.randint(23,(2,5));labels=ids.clone();labels[0,2]=-100
    a=reference(ids,labels=labels,use_cache=False);b=model(ids,labels=labels,use_cache=False)
    torch.testing.assert_close(a.logits,b.logits,atol=0,rtol=0)
    torch.testing.assert_close(a.loss,b.loss)
    (a.loss*.37).backward();(b.loss*.37).backward()
    for a,b in zip(reference.parameters(),model.parameters()):torch.testing.assert_close(a.grad,b.grad)
    torch.testing.assert_close(reference(ids,use_cache=False).logits,model(ids,use_cache=False).logits,atol=0,rtol=0)
    with pytest.raises(ValueError):install_ce_only_qwen_loss(model,chunk_size=3)


def test_ce_only_not_silently_mixed_into_old_screen():
    r=result();r['optimization']['ce_only_chunk_size']=128
    with pytest.raises(ValueError):audit_result(r,(24,0,'full'))
    audit_result(r,(24,0,'full'),allow_ce_only=True)
    from training.fsdp_search import workload_signature
    original=result()
    assert workload_signature(original)!=workload_signature(r)
