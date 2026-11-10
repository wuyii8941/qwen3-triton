import hashlib
import json

import pytest

from scripts import audit_fsdp_exchange_expandable_preflights as module


def fixture(root, monkeypatch):
    monkeypatch.chdir(root)
    (root / 'training').mkdir()
    original = {'validation': {'every_updates': 32}}
    (root / 'training/fsdp_convergence_protocol_v1.json').write_text(json.dumps(original))
    original['validation']['every_updates'] = 2
    (root / 'post_update_protocol.json').write_text(json.dumps(original))
    path = root / 'tiny_native'
    path.mkdir()
    value = {'validation': [{'step': 0}, {'step': 1}, {'step': 2}]}
    raw = json.dumps(value).encode()
    (path / 'result.json').write_bytes(raw)
    (root / 'tiny_native.log').write_text('synthetic')
    (root / 'tiny_native_resources.json').write_text(json.dumps({'samples': [
        {'mem_available_bytes': 160 * 2**30, 'disk_free_bytes': 16 * 2**30}]}))
    row = dict(case='tiny_native', source=str(path / 'result.json'), sha256=hashlib.sha256(raw).hexdigest(),
        log_sha256=hashlib.sha256(b'synthetic').hexdigest(), manifest_sha256=['a'] * 4,
        allocator_sha256=['b'] * 4, validation=value['validation'])
    (root / 'progress.json').write_text(json.dumps([row]))
    monkeypatch.setattr(module, 'audit_job', lambda *a, **kw: (value, ['a'] * 4))
    monkeypatch.setattr(module, 'audit_allocator', lambda *a: ['b'] * 4)
    monkeypatch.setattr(module, 'audit_launch', lambda *a, **kw: 'c' * 64)


def test_partial_cannot_prove_oom_fix(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        module.audit(tmp_path)
    r = module.audit(tmp_path, allow_partial=True)
    assert r['raw_records_audited'] == 1
    assert not r['long_run_oom_fixed_proven'] and not r['all_preflights_passed']


@pytest.mark.parametrize('field,value', [('allocator_sha256', []), ('manifest_sha256', []),
    ('sha256', 'wrong'), ('case', 'real_decoder'), ('validation', [])])
def test_corrupt_evidence_rejected(tmp_path, monkeypatch, field, value):
    fixture(tmp_path, monkeypatch)
    path = tmp_path / 'progress.json'
    rows = json.loads(path.read_text())
    rows[0][field] = value
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        module.audit(tmp_path, allow_partial=True)
