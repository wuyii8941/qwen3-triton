from types import SimpleNamespace

import pytest

from benchmarks.fsdp_exchange_quality import configure, install


def clean(monkeypatch):
    for name in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE'):
        monkeypatch.delenv(name, raising=False)


BASE = ['--backend', 'official', '--output', 'unused', '--memory-budget-gib', '42']


def test_native_and_forwarded_preflight(monkeypatch):
    clean(monkeypatch)
    args, base = configure(BASE + ['--tiny', '--preflight-steps', '2'])
    assert args.exchange_backend is None and '--tiny' in base
    assert '--memory-budget-gib' not in base


def test_decoder_and_head_environment(monkeypatch):
    clean(monkeypatch)
    configure(BASE + ['--exchange-backend', 'torch', '--exchange-scope', 'decoder'])
    with pytest.raises(ValueError):
        configure(BASE + ['--exchange-backend', 'triton', '--exchange-scope', 'head'])
    monkeypatch.setenv('NCCL_PROTO', 'Simple')
    configure(BASE + ['--exchange-backend', 'triton', '--exchange-scope', 'head'])
    with pytest.raises(ValueError):
        configure(BASE + ['--exchange-backend', 'torch', '--exchange-scope', 'decoder'])


@pytest.mark.parametrize('extra', [['--exchange-backend', 'torch'],
    ['--exchange-backend', 'torch', '--exchange-scope', 'head'],
    ['--bounded-exchange-scope', 'decoder'], ['--exchange-reduce', 'torch']])
def test_wrong_modes_rejected(monkeypatch, extra):
    clean(monkeypatch)
    with pytest.raises(ValueError):
        configure(BASE + extra)


class Unit:
    def __init__(self):
        self.weight = object()
    def set_custom_reduce_scatter(self, callback):
        self.callback = callback


def model():
    return SimpleNamespace(model=SimpleNamespace(layers=[Unit(), Unit()], embed_tokens=Unit()), lm_head=Unit())


def test_decoder_units_and_debug_mode():
    value = model()
    callbacks = install(value, 'torch', 'decoder', fsdp_type=Unit)
    assert [n for n, _ in callbacks] == ['model.layers.0', 'model.layers.1']
    assert all(not c.debug and c.backend == 'torch' for _, c in callbacks)
    assert not hasattr(value.lm_head, 'callback')


def test_head_only_and_tied_rejection():
    value = model()
    callbacks = install(value, 'triton', 'head', fsdp_type=Unit)
    assert len(callbacks) == 1 and callbacks[0][0] == 'lm_head'
    assert not hasattr(value.model.layers[0], 'callback')
    value.lm_head.weight = value.model.embed_tokens.weight
    with pytest.raises(ValueError):
        install(value, 'triton', 'head', fsdp_type=Unit)


def test_native_and_unsharded_rejection():
    value = model()
    assert install(value, None, None, fsdp_type=Unit) == []
    with pytest.raises(ValueError):
        install(value, 'torch', 'decoder', fsdp_type=dict)
