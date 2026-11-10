import copy
import pytest
import torch
from training.fsdp_accumulation import (step_sample_indices,sync_this_microbatch,gradient_sync,
    shifted_valid_tokens,global_weighted_causal_loss)


def test_batch_repartition_preserves_exact_rank_samples():
    for step in (0,3):
        for rank in range(4):
            large=step_sample_indices(step,rank,4,4,1)
            split=step_sample_indices(step,rank,4,1,4)
            assert large[0]==sum(split,[])
            assert step_sample_indices(step,rank,4,2,1)[0]==[(step*4+rank)*2+j for j in range(2)]


def test_sync_schedule_and_exception_cleanup():
    assert [sync_this_microbatch(i,3,'each') for i in range(3)]==[True]*3
    assert [sync_this_microbatch(i,3,'defer') for i in range(3)]==[False,False,True]
    class Fake:
        def __init__(self):self.flags=[]
        def set_requires_gradient_sync(self,enabled):self.flags.append(enabled)
    model=Fake()
    with pytest.raises(RuntimeError):
        with gradient_sync(model,False):raise RuntimeError('backward failed')
    assert model.flags==[False,True]
    with pytest.raises(ValueError):
        with gradient_sync(object(),False):pass


def test_uneven_masks_and_empty_local_microbatch_match_global_token_mean():
    torch.manual_seed(17)
    model=torch.nn.Linear(5,7);reference=copy.deepcopy(model)
    x=torch.randn(4,6,5);labels=torch.randint(7,(4,6))
    labels[0]=-100;labels[1,3:]=-100;labels[2,2:]=-100
    count=int(shifted_valid_tokens(labels))
    expected=global_weighted_causal_loss(reference(x),labels,global_tokens=count,world=1)
    expected.backward()
    losses=[]
    for i in range(4):
        part=global_weighted_causal_loss(model(x[i:i+1]),labels[i:i+1],global_tokens=count,world=2)
        (part/2).backward();losses.append(part.detach()/2)
    torch.testing.assert_close(sum(losses),expected.detach())
    for a,b in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)


@pytest.mark.parametrize('which',['rank','shape','empty','sync'])
def test_invalid_config_rejected(which):
    with pytest.raises(ValueError):
        if which=='rank':step_sample_indices(0,4,4,1,2)
        elif which=='shape':shifted_valid_tokens(torch.zeros(2,1,dtype=torch.int64))
        elif which=='empty':global_weighted_causal_loss(torch.zeros(1,2,3),torch.zeros(1,2,dtype=torch.int64),global_tokens=0,world=1)
        else:sync_this_microbatch(0,2,'silent_fallback')
