"""CPU scheduling checks only; NOT a substitute for GPU gradient parity."""
from types import SimpleNamespace
import torch
import pytest
import training.prefetch_fsdp as module


class Buffer:
    def record_stream(self,stream):pass


class Work:
    def wait(self):pass


class Stream:
    def wait_stream(self,stream):pass


class Unit:
    def __init__(self,index):
        self.index=index;self.size=4;self.world=2;self.compute_dtype=torch.float32
        self.shard=SimpleNamespace(device='cpu')


@pytest.mark.parametrize('phase',['forward','backward'])
def test_collective_order_dedup_and_step_reset(monkeypatch,phase):
    from contextlib import nullcontext
    calls=[]
    monkeypatch.setattr(module.torch.cuda,'Stream',Stream)
    monkeypatch.setattr(module.torch.cuda,'current_stream',lambda:'compute')
    monkeypatch.setattr(module.torch.cuda,'stream',lambda stream:nullcontext())
    monkeypatch.setattr(module.torch,'empty',lambda *a,**k:Buffer())
    monkeypatch.setattr(module,'copy_scale',lambda *a,**k:Buffer())
    monkeypatch.setattr(module.dist,'all_gather_into_tensor',lambda *a,**k:(calls.append('gather') or Work()))
    units=[Unit(i) for i in range(3)]
    engine=module.ParameterPrefetch(units)
    order=units if phase=='forward' else list(reversed(units))
    for _ in range(2):
        engine.request(order[0],order[0].shard,phase)
        engine.request(order[0],order[0].shard,phase)
        before=len(calls)
        for unit in order:engine.acquire(unit,unit.shard,phase)
        assert len(calls)-before==2
        assert not engine.pending
    assert len(calls)==6
