import pytest
import torch
import torch.distributed as dist
from training.fsdp.communication import BF16ExchangeReduceScatter


@pytest.mark.parametrize('kwargs', [dict(original_bf16=False),
    dict(original_bf16=True, accumulation=2), dict(original_bf16=True, backend='bad')])
def test_reject_unsupported_contract(kwargs):
    with pytest.raises(ValueError):
        BF16ExchangeReduceScatter(**kwargs)


def test_allocate_honors_requested_dtype():
    callback = BF16ExchangeReduceScatter(original_bf16=True)
    for dtype in (torch.float32, torch.bfloat16):
        tensor = callback.allocate((3, 4), dtype=dtype, device='cpu')
        assert tensor.shape == (3, 4) and tensor.dtype == dtype


def test_async_is_explicitly_rejected():
    callback = BF16ExchangeReduceScatter(original_bf16=True)
    with pytest.raises(ValueError, match='Async'):
        callback(None, None, None, dist.ReduceOp.AVG, async_op=True)


@pytest.mark.parametrize('average', [False, True])
def test_callback_transport_and_output(monkeypatch, average):
    monkeypatch.setattr(dist, 'get_world_size', lambda group: 2)
    monkeypatch.setattr(dist, 'all_reduce', lambda value, **kwargs: None)
    def exchange(out, inp, **kwargs):
        out.copy_(torch.tensor([2., -4., 6., 8., 1., 2.], dtype=torch.bfloat16))
    monkeypatch.setattr(dist, 'all_to_all_single', exchange)
    callback = BF16ExchangeReduceScatter(original_bf16=True)
    out = torch.empty(3)
    inp = torch.arange(6, dtype=torch.float32)
    callback(out, inp, None, dist.ReduceOp.AVG if average else dist.ReduceOp.SUM)
    torch.testing.assert_close(out, torch.tensor([10., -3., 8.]) / (2 if average else 1))
    assert callback.calls == 1


def test_lossy_fp32_is_rejected_before_exchange(monkeypatch):
    monkeypatch.setattr(dist, 'get_world_size', lambda group: 2)
    monkeypatch.setattr(dist, 'all_reduce', lambda value, **kwargs: None)
    callback = BF16ExchangeReduceScatter(original_bf16=True)
    with pytest.raises(ValueError, match='exact'):
        callback(torch.empty(2), torch.tensor([.1234567]*4), None, dist.ReduceOp.AVG)
