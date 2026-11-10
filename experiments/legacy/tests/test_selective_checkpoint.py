import copy
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from torch.nn.attention import sdpa_kernel,SDPBackend
from training.selective_checkpoint import enable_selective_checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_selective_checkpoint_compiled_gradients(dtype):
    torch.manual_seed(61)
    cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=True)
    cfg._attn_implementation='sdpa'
    a=Qwen3ForCausalLM(cfg).to(device='cuda',dtype=dtype).train()
    b=enable_selective_checkpoint(copy.deepcopy(a))
    def fn(model):return torch.compile(lambda ids:model(ids,labels=ids,use_cache=False).loss,fullgraph=True,dynamic=False)
    fa,fb=fn(a),fn(b);ids=torch.randint(0,97,(2,15),device='cuda')
    with sdpa_kernel(SDPBackend.MATH):
        la,lb=fa(ids),fb(ids);la.backward();lb.backward()
    torch.testing.assert_close(la,lb,rtol=.005 if dtype==torch.bfloat16 else 1e-5,atol=1e-6)
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        rel=(p.grad.float()-q.grad.float()).norm()/p.grad.float().norm().clamp_min(1e-12)
        assert float(rel)<(.025 if dtype==torch.bfloat16 else 5e-5),(name,float(rel))
    assert a.state_dict().keys()==b.state_dict().keys()
