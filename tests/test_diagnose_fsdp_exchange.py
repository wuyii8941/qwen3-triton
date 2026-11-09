import hashlib
import json
from pathlib import Path

import pytest

from scripts.diagnose_fsdp_exchange import CASES, command, conclusions
from scripts import diagnose_fsdp_exchange as module


def runs():
    return {name: {'records': [{'loss': v} for v in (2.65, 2.68, 2.73, 2.30)]} for name in CASES}


def test_matched_retention_and_short_scope():
    for name, (backend, scope, _) in CASES.items():
        cmd = command(name, Path('unused'))
        for flag, value in (('--keep-full-layers', '32'), ('--memory-budget-gib', '42'),
                            ('--warmup', '1'), ('--steps', '3'), ('--checkpoint-every', '1')):
            assert cmd[cmd.index(flag) + 1] == value
        assert '--reference-check' in cmd
        assert ('--exchange-reduce' in cmd) == bool(backend)
        if backend:
            assert cmd[cmd.index('--exchange-reduce') + 1] == backend
            assert cmd[cmd.index('--exchange-scope') + 1] == scope


def test_repeat_and_candidate_checks():
    checks = conclusions(runs())
    assert all(c['native_repeat_exact'] and c['eligible_for_quality_preflight'] for c in checks.values())


@pytest.mark.parametrize('native,candidate', [('auto_b', 'decoder'), ('simple_b', 'head')])
def test_noisy_native_repeat_never_qualifies(native, candidate):
    values = runs()
    values[native]['records'][0]['loss'] += 1e-7
    result = conclusions(values)[candidate]
    assert not result['native_repeat_exact']
    assert not result['eligible_for_quality_preflight']


@pytest.mark.parametrize('candidate', ['decoder', 'head'])
def test_candidate_error_does_not_relax_threshold(candidate):
    values = runs()
    values[candidate]['records'][1]['loss'] += .01
    assert not conclusions(values)[candidate]['eligible_for_quality_preflight']


def test_partial_never_qualifies():
    values = runs()
    values.pop('head')
    with pytest.raises(ValueError):
        conclusions(values)
    values = runs()
    values['decoder']['records'].pop()
    with pytest.raises(ValueError):
        conclusions(values)


def fixture_directory(tmp_path, monkeypatch):
    sources = {'benchmarks/deterministic_fsdp_comparison.py': 'a' * 64,
               'benchmarks/compare_fsdp_training.py': 'b' * 64}
    snapshot = {'sha256': 'c' * 64}
    monkeypatch.setattr(module, 'hashes', lambda: sources)
    monkeypatch.setattr(module, 'source_snapshot', lambda root: snapshot)
    monkeypatch.setattr(module, 'audit_result', lambda *a, **kw: ('synthetic',))
    env = {key: None for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE')}
    result = dict(global_parameters=8190735360, world_size=4, seq=512, batch=1,
        warmup=1, steps=3, memory_budget_gib=42,
        optimization={'exchange_reduce': 'torch', 'exchange_scope': 'decoder'},
        communication_environment=env,
        deterministic_diagnostic=dict(enabled=True, cublas_workspace=':4096:8',
            source_hashes={Path(n).name: h for n, h in sources.items()}),
        manifest=dict(input_sha256_by_rank=['d' * 64] * 4, implementation_sources=snapshot),
        exchange_callback_counts_by_rank=[{f'model.layers.{i}': 4 for i in range(36)} for _ in range(4)])
    raw = json.dumps(result).encode()
    (tmp_path / 'result.json').write_bytes(raw)
    for rank in range(4):
        context = dict(rank=rank, algorithms_enabled=True, warn_only=False,
            cublas_workspace=':4096:8', communication_environment=env, source_sha256=sources)
        if rank == 0:
            context['result_sha256'] = hashlib.sha256(raw).hexdigest()
        (tmp_path / f'exchange_deterministic_rank{rank}.json').write_text(json.dumps(context))


def test_all_rank_evidence_audit(tmp_path, monkeypatch):
    fixture_directory(tmp_path, monkeypatch)
    _, _, contexts = module.audit_short(tmp_path, 'decoder')
    assert len(contexts) == 4


@pytest.mark.parametrize('field,value', [('rank', 0), ('warn_only', True),
    ('algorithms_enabled', False), ('cublas_workspace', None), ('source_sha256', {})])
def test_bad_rank_context_rejected(tmp_path, monkeypatch, field, value):
    fixture_directory(tmp_path, monkeypatch)
    path = tmp_path / 'exchange_deterministic_rank2.json'
    context = json.loads(path.read_text())
    context[field] = value
    path.write_text(json.dumps(context))
    with pytest.raises(ValueError):
        module.audit_short(tmp_path, 'decoder')
