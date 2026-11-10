import pytest
import torch
from training.fsdp_checkpoint import required_bytes,schema,_canonical,_trainer_state


def test_disk_estimate_counts_global_weights_and_two_fp32_moments():
    m=torch.nn.Linear(3,2)
    assert required_bytes(m)==int(8*12*1.1)+16*2**20
    assert schema(m)==[['weight',[2,3],'torch.float32'],['bias',[2],'torch.float32']]


def test_reject_non_fp32_and_unsupported_optimizer():
    m=torch.nn.Linear(3,2)
    with pytest.raises(ValueError):_canonical(m,torch.optim.SGD(m.parameters(),lr=.1))
    with pytest.raises(ValueError):_canonical(m,torch.optim.AdamW(m.parameters(),amsgrad=True))
    m=m.to(torch.bfloat16)
    with pytest.raises(ValueError):_canonical(m,torch.optim.AdamW(m.parameters()))


def test_cpu_state_does_not_depend_on_incidental_cuda_initialization(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda:pytest.fail('CPU model must not read CUDA RNG'))
    assert _trainer_state(2,{'next_block':2},uses_cuda=False)['cuda_rng'] is None
