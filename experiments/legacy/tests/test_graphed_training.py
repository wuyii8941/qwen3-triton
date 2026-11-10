import copy
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from torch.nn.attention import SDPBackend,sdpa_kernel
from training.graphed_forward_backward import GraphedForwardBackward
from benchmarks.check_tied_embedding import patch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('embedding',[False,True,'consumed'])
def test_replay_changes_input_overwrites_grad_and_updates_parameters(embedding):
    torch.manual_seed(88)
    cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                   num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=True)
    cfg._attn_implementation='sdpa'
    reference=Qwen3ForCausalLM(cfg).cuda().train()
    candidate=copy.deepcopy(reference)
    if embedding:
        patch(reference,consume_proxy=embedding=='consumed');patch(candidate,consume_proxy=embedding=='consumed')
    def make_fn(model):
        return torch.compile(lambda x:model(x,labels=x,use_cache=False).loss,fullgraph=True,dynamic=False)
    rf,cf=make_fn(reference),make_fn(candidate)
    ro=torch.optim.AdamW(reference.parameters(),lr=1e-4,fused=True)
    co=torch.optim.AdamW(candidate.parameters(),lr=1e-4,fused=True)
    x=torch.randint(0,97,(1,15),device='cuda')
    with sdpa_kernel(SDPBackend.MATH):
        runner=GraphedForwardBackward(candidate,cf,x)
        for _ in range(4):
            # New data and updated weights must both affect the next replay.
            x=torch.randint(0,97,(1,15),device='cuda')
            ro.zero_grad(set_to_none=True)
            rl=rf(x);rl.backward()
            cl=runner.backward(x)
            torch.testing.assert_close(rl,cl,rtol=1e-5,atol=1e-5)
            for a,b in zip(reference.parameters(),candidate.parameters()):
                torch.testing.assert_close(a.grad,b.grad,rtol=2e-4,atol=2e-6)
            ro.step();co.step()
            for a,b in zip(reference.parameters(),candidate.parameters()):
                torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        candidate.eval()
        with pytest.raises(RuntimeError,match='model.train'):runner.backward(x)
        candidate.train()
        with pytest.raises(ValueError,match='shape/dtype/device'):runner.backward(x[:,:-1])
        p=next(candidate.parameters());old=p.detach()
        p.data=p.data.clone()
        with pytest.raises(RuntimeError,match='parameter buffers'):runner.backward(x)
        p.data=old
        candidate.zero_grad(set_to_none=True)
        with pytest.raises(RuntimeError,match='buffers replaced'):runner.backward(x)
