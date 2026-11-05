import copy
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from torch.nn.attention import sdpa_kernel,SDPBackend
from training.owned_tied_loss import OwnedTiedLoss


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('compile_regions',[False,True])
def test_owned_loss_gradients_accumulation_and_update(dtype,compile_regions):
    torch.manual_seed(59)
    cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=True)
    cfg._attn_implementation='sdpa'
    reference=Qwen3ForCausalLM(cfg).to(device='cuda',dtype=dtype).train()
    candidate=copy.deepcopy(reference);owned=OwnedTiedLoss(candidate,compile_regions=compile_regions)
    for _ in range(2):
        before=[p.detach().clone() for p in candidate.parameters()]
        with sdpa_kernel(SDPBackend.MATH):
            for _ in range(2):
                ids=torch.randint(0,97,(2,15),device='cuda');labels=ids.clone();labels[:,4]=-100
                a=reference(ids,labels=labels,use_cache=False).loss/2
                b=owned(ids,labels)/2
                torch.testing.assert_close(a,b,rtol=.005 if dtype==torch.bfloat16 else 1e-5,atol=1e-6)
                a.backward();b.backward()
        for snapshot,(name,p),(other,q) in zip(before,reference.named_parameters(),candidate.named_parameters()):
            assert name==other and p.grad is not None and q.grad is not None
            assert torch.equal(snapshot,q.detach()),'Backward must not mutate a model parameter'
            rel=(p.grad.float()-q.grad.float()).norm()/p.grad.float().norm().clamp_min(1e-12)
            assert float(rel)<(.025 if dtype==torch.bfloat16 else 5e-5),(name,float(rel))
            with torch.no_grad():p.add_(p.grad,alpha=-.001);q.add_(q.grad,alpha=-.001)
        reference.zero_grad(set_to_none=True);candidate.zero_grad(set_to_none=True)
    assert candidate.model.embed_tokens.weight is candidate.lm_head.weight
    assert candidate.state_dict().keys()==reference.state_dict().keys()
