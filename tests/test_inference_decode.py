from types import SimpleNamespace

import pytest
import torch

from qwen3_triton.inference import kernels


@pytest.fixture
def launches(monkeypatch):
    calls = []

    class Capture:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((self.name, grid, args, kwargs))
            return launch

    def allocate(shape, *, device, dtype):
        return SimpleNamespace(shape=shape, device=device, dtype=dtype)

    monkeypatch.setattr(kernels, "raw_empty", allocate)
    monkeypatch.setattr(kernels, "_inference_cached_attention", Capture("cached"))
    monkeypatch.setattr(kernels, "_inference_decode_partial", Capture("partial"))
    monkeypatch.setattr(kernels, "_inference_decode_merge", Capture("merge"))
    kernels._sm86_decode_device.cache_clear()
    yield calls
    kernels._sm86_decode_device.cache_clear()


def operands(batch, tokens, device=None):
    q = SimpleNamespace(shape=(batch, tokens, 32, 128), dtype=torch.bfloat16,
                        device=device or torch.device("cuda:0"))
    cache = SimpleNamespace(shape=(batch, 8, 4096, 128))
    return q, cache


@pytest.mark.parametrize(
    ("batch", "tokens", "keys", "capability", "compact", "rows"),
    [
        (1, 1, 512, (8, 6), True, 16),
        (1, 1, 2048, (8, 6), True, 16),
        (1, 1, 511, (8, 6), True, 64),
        (2, 1, 512, (8, 6), True, 64),
        (1, 2, 512, (8, 6), True, 64),
        (1, 1, 512, (8, 0), True, 64),
        (1, 1, 2048, (8, 6), False, 64),
    ],
)
def test_attention_launch_geometry(monkeypatch, launches, batch, tokens, keys, capability, compact, rows):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    q, cache = operands(batch, tokens)
    options = {"compact_decode": True} if compact else {}
    result = kernels.attention(q, cache, cache, keys - tokens, **options)

    [(name, grid, args, kwargs)] = launches
    assert name == "cached" and grid == (1, batch * 32)
    assert args[:4] == (q, cache, cache, result)
    assert args[4:7] == (tokens, keys, 4096)
    assert args[-2:] == (rows, 32 if tokens == 1 else 64)
    assert kwargs == dict(num_warps=4, num_stages=2, enable_fp_fusion=False)


def test_explicit_split_decode_keeps_two_stage_route(monkeypatch, launches):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    q, cache = operands(1, 1)
    result = kernels.attention(q, cache, cache, 2047, split_decode=True, compact_decode=True)
    assert result.shape == q.shape
    assert [(name, grid) for name, grid, _, _ in launches] == [("partial", (32, 8)), ("merge", (32,))]


def test_repeated_long_decode_reuses_capability_per_device(monkeypatch, launches):
    capability_checks = []

    def capability(device):
        capability_checks.append(device)
        return (8, 6)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    for device in devices * 2:
        q, cache = operands(1, 1, device)
        kernels.attention(q, cache, cache, 511, compact_decode=True)

    assert capability_checks == devices
