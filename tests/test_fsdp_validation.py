import torch
import torch.nn.functional as F
import pytest
from training.fsdp_validation import validation_batches,shifted_nll


@pytest.mark.parametrize('size,world,batch',[(1,4,4),(17,4,4),(32,4,4),(7,2,3)])
def test_global_validation_matches_unpartitioned(size,world,batch):
    torch.manual_seed(42)
    ids=torch.randint(0,11,(size,5))
    table=torch.randn(size,5,11)
    total,count=0.,0
    forwards=[]
    for rank in range(world):
        calls=0
        for data,valid in validation_batches(list(ids),rank,world,batch):
            start=calls*world*batch+rank*batch
            logits=torch.stack([table[i] if i<size else table[0] for i in range(start,start+batch)])
            n,c=shifted_nll(logits,data,valid)
            assert torch.isfinite(n)
            total+=n.item();count+=c.item();calls+=1
        forwards.append(calls)
    expected=F.cross_entropy(table[:,:-1].reshape(-1,11),ids[:,1:].reshape(-1),reduction='sum')
    assert len(set(forwards))==1
    assert count==size*4
    assert total==pytest.approx(expected.item(),rel=1e-6)
