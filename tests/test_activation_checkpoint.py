import copy
import pytest
import torch
from training.fsdp.activation_checkpoint import checkpoint_layer_indices,wrap_decoder_checkpoint,canonical_parameter_name


def test_layer_selection_and_ambiguous_config_rejected():
    assert checkpoint_layer_indices(5)==()
    assert checkpoint_layer_indices(5,every=2)==(0,2,4)
    assert checkpoint_layer_indices(5,all_layers=True)==(0,1,2,3,4)
    for kwargs in (dict(every=-1),dict(every=True),dict(every=2,all_layers=True)):
        with pytest.raises(ValueError):checkpoint_layer_indices(5,**kwargs)
    with pytest.raises(ValueError):checkpoint_layer_indices(0)


@pytest.mark.parametrize('mode',['full'])
def test_cpu_checkpoint_preserves_dropout_rng_gradients_and_state_dict(mode):
    torch.manual_seed(51)
    reference=torch.nn.Sequential(torch.nn.Linear(7,11),torch.nn.Dropout(.3),torch.nn.Linear(11,5))
    wrapped=wrap_decoder_checkpoint(copy.deepcopy(reference),mode=mode)
    a=torch.optim.AdamW(reference.parameters(),lr=.001)
    b=torch.optim.AdamW(wrapped.parameters(),lr=.001)
    x=torch.randn(4,7)
    for step in range(3):
        a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
        torch.manual_seed(100+step)
        expected=reference(x).square().mean();expected.backward()
        expected_rng=torch.get_rng_state()
        torch.manual_seed(100+step)
        actual=wrapped(x).square().mean();actual.backward()
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        torch.testing.assert_close(torch.get_rng_state(),expected_rng,atol=0,rtol=0)
        for p,q in zip(reference.parameters(),wrapped.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
        a.step();b.step()
        assert reference.state_dict().keys()==wrapped.state_dict().keys()
        for name,param in wrapped.named_parameters():
            torch.testing.assert_close(param,reference.get_parameter(canonical_parameter_name(name)),atol=0,rtol=0)


def test_nested_wrapper_parameter_names_map_to_reference():
    assert canonical_parameter_name('model.layers.0._checkpoint_wrapped_module.q_proj.weight')=='model.layers.0.q_proj.weight'


@pytest.mark.parametrize('mode',['full'])
def test_tiny_qwen_cpu_loss_gradients_and_updates(mode):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(71)
    config=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        attention_dropout=0.,tie_word_embeddings=True)
    config._attn_implementation='sdpa'
    reference=Qwen3ForCausalLM(config).train();wrapped=copy.deepcopy(reference)
    wrapped.model.layers[0]=wrap_decoder_checkpoint(wrapped.model.layers[0],mode=mode)
    a=torch.optim.AdamW(reference.parameters(),lr=.001)
    b=torch.optim.AdamW(wrapped.parameters(),lr=.001)
    for step in range(2):
        ids=torch.randint(97,(2,11))
        a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
        expected=reference(ids,labels=ids,use_cache=False)
        actual=wrapped(ids,labels=ids,use_cache=False)
        torch.testing.assert_close(actual.logits,expected.logits,atol=0,rtol=0)
        torch.testing.assert_close(actual.loss,expected.loss,atol=0,rtol=0)
        expected.loss.backward();actual.loss.backward()
        for name,param in wrapped.named_parameters():
            target=reference.get_parameter(canonical_parameter_name(name))
            torch.testing.assert_close(param.grad,target.grad,atol=1e-7,rtol=1e-6)
        a.step();b.step()
        for name,param in wrapped.named_parameters():
            target=reference.get_parameter(canonical_parameter_name(name))
            torch.testing.assert_close(param,target,atol=1e-7,rtol=1e-6)
        assert wrapped.state_dict().keys()==reference.state_dict().keys()
