from pathlib import Path
import hashlib
import json

import pytest

from benchmarks.fsdp_exchange_quality_expandable import ALLOCATOR, EFFECTIVE_CAP_GIB, validate_settings, effective_fraction
from scripts.preflight_fsdp_exchange_expandable import command, environment
from scripts import preflight_fsdp_exchange_expandable as module


def test_actual_allocator_settings_not_just_environment():
    good = {'expandable_segments': True, 'PYTORCH_CUDA_ALLOC_CONF': ALLOCATOR}
    validate_settings(good, 'native', ALLOCATOR, None)
    for value in ({}, dict(good, expandable_segments=False)):
        with pytest.raises(ValueError):
            validate_settings(value, 'native', ALLOCATOR, None)
    for backend, configured, legacy in [('cudaMallocAsync', ALLOCATOR, None), ('native', 'wrong', None), ('native', ALLOCATOR, 'conflict')]:
        with pytest.raises(ValueError):
            validate_settings(good, backend, configured, legacy)


def test_effective_cap_is_lower_not_relaxed():
    total = 48 * 2**30
    assert effective_fraction(42 / 48, total) * 48 == EFFECTIVE_CAP_GIB < 42
    with pytest.raises(ValueError):
        effective_fraction(43 / 48, total)


@pytest.mark.parametrize('mode,keep', [('native', '36'), ('decoder', '32')])
@pytest.mark.parametrize('preflight', [True, False])
def test_identical_budgets_and_explicit_new_worker(mode, keep, preflight):
    cmd = command(Path('unused'), mode, Path('protocol'), preflight=preflight)
    assert 'benchmarks.fsdp_exchange_quality_expandable' in cmd
    assert cmd[cmd.index('--memory-budget-gib') + 1] == '42'
    assert cmd[cmd.index('--keep-full-layers') + 1] == keep
    assert ('--preflight-steps' in cmd) == preflight


def test_environment_clears_conflicts_and_preserves_host(monkeypatch):
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'old')
    monkeypatch.setenv('NCCL_PROTO', 'Simple')
    env = environment()
    assert env['PYTORCH_ALLOC_CONF'] == ALLOCATOR
    assert 'PYTORCH_CUDA_ALLOC_CONF' not in env and 'NCCL_PROTO' not in env
    import os
    assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'old'


def test_no_tiny_without_cutoff():
    with pytest.raises(ValueError):
        command(Path('unused'), 'native', Path('protocol'), tiny=True, preflight=False)


def fixture(root, monkeypatch):
    sources = {'worker': 'a' * 64}
    monkeypatch.setattr(module, 'worker_hashes', lambda: sources)
    (root / 'result.json').write_text('{}')
    good = {'expandable_segments': True, 'PYTORCH_CUDA_ALLOC_CONF': ALLOCATOR}
    for rank in range(4):
        (root / f'exchange_quality_rank{rank}.json').write_text('{}')
        row = dict(rank=rank, passed=True, backend='native', configured=ALLOCATOR,
            audited_reserved_upper_bound_gib=42., effective_allocator_cap_gib=EFFECTIVE_CAP_GIB,
            actual_memory_fraction=EFFECTIVE_CAP_GIB / 48, device_total_bytes=48 * 2**30,
            actual_settings_before=good, actual_settings_after=good, source_sha256=sources,
            base_manifest_sha256=hashlib.sha256(b'{}').hexdigest())
        if rank == 0:
            row['result_sha256'] = hashlib.sha256(b'{}').hexdigest()
        (root / f'expandable_allocator_rank{rank}.json').write_text(json.dumps(row))


def test_all_rank_actual_settings_audit(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    assert len(module.audit_allocator(tmp_path)) == 4


@pytest.mark.parametrize('field,value', [('rank', 0), ('source_sha256', {}),
    ('base_manifest_sha256', 'wrong'), ('actual_settings_after', {'expandable_segments': False})])
def test_bad_allocator_proof_rejected(tmp_path, monkeypatch, field, value):
    fixture(tmp_path, monkeypatch)
    path = tmp_path / 'expandable_allocator_rank2.json'
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        module.audit_allocator(tmp_path)
