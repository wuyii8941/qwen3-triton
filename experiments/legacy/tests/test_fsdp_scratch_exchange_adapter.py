import gc
import weakref

import pytest
import torch

import benchmarks.fsdp_scratch_exchange_adapter as adapter


@pytest.mark.parametrize('options', [
    dict(original_bf16=False), dict(original_bf16=True, accumulation=2),
    dict(original_bf16=True, backend='torch'), dict(original_bf16=True, debug=False),
])
def test_unsupported_modes_rejected(options):
    with pytest.raises(ValueError):
        adapter.ScratchExchangeReduceScatter(**options)


def test_unpinned_torch_rejected(monkeypatch):
    monkeypatch.setattr(torch, '__version__', 'untested')
    with pytest.raises(ValueError, match='pinned'):
        adapter.verify_private_contract()


def test_changed_private_source_rejected(monkeypatch):
    monkeypatch.setitem(adapter.PINNED, '_fsdp_collectives.py', 'not_the_tested_hash')
    with pytest.raises(ValueError, match='lifetime source'):
        adapter.verify_private_contract()


def test_no_cpu_or_bf16_allocation():
    instance = adapter.ScratchExchangeReduceScatter(original_bf16=True)
    for dtype, device in ((torch.float32, 'cpu'), (torch.bfloat16, 'cuda')):
        with pytest.raises(ValueError, match='FP32 CUDA'):
            instance.allocate((8,), dtype=dtype, device=device)


def test_registry_does_not_retain_scratch():
    instance = adapter.ScratchExchangeReduceScatter(original_bf16=True)
    tensor = torch.ones(8)
    key, ref = id(tensor), weakref.ref(tensor)
    instance.owned[key] = tensor
    del tensor
    gc.collect()
    assert ref() is None and key not in instance.owned
