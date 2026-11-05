import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from training.hf_operators import training_operators


def test_scoped_training_patch_restores_on_exception():
    c=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16)
    m=Qwen3ForCausalLM(c)
    modules=[x for x in m.modules() if x.__class__.__name__ in ('Qwen3RMSNorm','Qwen3MLP')]
    assert modules and all('forward' not in x.__dict__ for x in modules)
    with pytest.raises(RuntimeError,match='intentional'):
        with training_operators(m,rms=True,swiglu=True):
            assert all('forward' in x.__dict__ for x in modules)
            raise RuntimeError('intentional')
    assert all('forward' not in x.__dict__ for x in modules)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('variant',['rms','swiglu','qk','operators'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_hf_loss_and_all_parameter_gradients(variant,dtype):
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.manual_seed(29)
    c=Qwen3Config(vocab_size=256,hidden_size=128,intermediate_size=384,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=64,
        attention_dropout=0.,rope_theta=1000000.)
    c._attn_implementation='sdpa'
    m=Qwen3ForCausalLM(c).cuda().to(dtype).train()
    ids=torch.randint(256,(2,31),device='cuda')
    # FP32 Flash is unsupported: math is used ONLY for the FP32 oracle test.
    backend=SDPBackend.FLASH_ATTENTION if dtype==torch.bfloat16 else SDPBackend.MATH
    with sdpa_kernel(backend):
        expected=m(ids,labels=ids,use_cache=False).loss
        expected.backward()
        grads={n:p.grad.detach().float().clone() for n,p in m.named_parameters()}
        m.zero_grad(set_to_none=True)
        with training_operators(m,rms=variant in ('rms','operators'),
                swiglu=variant in ('swiglu','operators'),qk=variant in ('qk','operators')):
            actual=m(ids,labels=ids,use_cache=False).loss
            actual.backward()
        torch.testing.assert_close(actual,expected,atol=.002,rtol=.002)
        errors={n:float((p.grad.float()-grads[n]).norm()/grads[n].norm().clamp_min(1e-12))
                for n,p in m.named_parameters()}
        bound=.03 if dtype==torch.bfloat16 else .0001
        assert max(errors.values())<bound,errors
