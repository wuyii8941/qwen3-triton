from types import SimpleNamespace
import pytest
import torch
from benchmarks.verify_fsdp_1p7b_states import tensor_record,named_parameters


def test_byte_hash_distinguishes_signed_zero_and_values():
    a=tensor_record(torch.tensor([0.,1.]))
    assert a['finite'] and a['dtype']=='torch.float32'
    assert a['sha256']!=tensor_record(torch.tensor([-0.,1.]))['sha256']
    assert a['sha256']!=tensor_record(torch.tensor([0.,2.]))['sha256']
    assert a==tensor_record(torch.tensor([0.,1.]))


def test_scalar_and_nonfinite():
    assert tensor_record(torch.tensor(3.))['shape']==[]
    assert not tensor_record(torch.tensor(float('nan')))['finite']
    assert not tensor_record(torch.tensor(float('inf')))['finite']


def test_global_and_local_shapes_both_recorded():
    class Shard:
        shape=(4,)
        def to_local(self):return torch.arange(2,dtype=torch.float32)
    record=tensor_record(Shard())
    assert record['shape']==[4] and record['local_shape']==[2]


def test_checkpoint_names_canonicalized_and_collisions_rejected():
    value=torch.tensor(1.)
    model=SimpleNamespace(named_parameters=lambda:iter([('x._checkpoint_wrapped_module.weight',value)]))
    assert list(named_parameters(model))==['x.weight']
    model.named_parameters=lambda:iter([('x._checkpoint_wrapped_module.weight',value),('x.weight',value)])
    with pytest.raises(ValueError,match='collision'):named_parameters(model)
