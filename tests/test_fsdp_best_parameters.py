import torch
import pytest
from training.fsdp_best_parameters import BestParameters


def test_reuses_cpu_storage_and_restores_best():
    model=torch.nn.Linear(3,2)
    best=BestParameters();best.save(model)
    pointers={n:v.data_ptr() for n,v in best.values.items()}
    expected={n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():p.add_(4)
    best.restore(model)
    for n,p in model.named_parameters():torch.testing.assert_close(p,expected[n])
    best.save(model)
    assert pointers=={n:v.data_ptr() for n,v in best.values.items()}
    with pytest.raises(ValueError):best.restore(torch.nn.Linear(4,2))
