import hashlib
import json

import pytest

from scripts import audit_fsdp_exchange_quality_preflights as module


def fixture(root, monkeypatch):
    monkeypatch.chdir(root)
    (root / 'training').mkdir()
    protocol = {'validation': {'every_updates': 32}, 'sequence_length': 512}
    (root / 'training/fsdp_convergence_protocol_v1.json').write_text(json.dumps(protocol))
    protocol['validation']['every_updates'] = 2
    (root / 'post_update_protocol.json').write_text(json.dumps(protocol))
    directory = root / 'tiny_native'
    directory.mkdir()
    result = {'validation': [{'step': 0}, {'step': 1}, {'step': 2}]}
    raw = json.dumps(result).encode()
    (directory / 'result.json').write_bytes(raw)
    (root / 'tiny_native.log').write_text('synthetic')
    (root / 'tiny_native_resources.json').write_text(json.dumps({'samples': [
        {'mem_available_bytes': 160 * 2**30, 'disk_free_bytes': 16 * 2**30}]}))
    row = dict(case='tiny_native', source=str(directory / 'result.json'),
        sha256=hashlib.sha256(raw).hexdigest(), log_sha256=hashlib.sha256(b'synthetic').hexdigest(),
        manifest_sha256=['a'] * 4, validation=result['validation'], best_parameters_restored=True)
    (root / 'progress.json').write_text(json.dumps([row]))
    monkeypatch.setattr(module, 'audit_job', lambda *a, **kw: (result, ['a'] * 4))
    monkeypatch.setattr(module, 'audit_launch', lambda *a, **kw: 'b' * 64)


def test_partial_cannot_claim_quality_or_complete_preflights(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        module.audit(tmp_path)
    result = module.audit(tmp_path, allow_partial=True)
    assert result['raw_records_audited'] == 1
    assert not result['complete'] and not result['all_preflights_passed'] and not result['full_quality_completed']


@pytest.mark.parametrize('fault', ['hash', 'order', 'validation', 'manifest', 'protocol', 'resources'])
def test_corrupt_partial_evidence_rejected(tmp_path, monkeypatch, fault):
    fixture(tmp_path, monkeypatch)
    if fault == 'protocol':
        (tmp_path / 'post_update_protocol.json').write_text('{}')
    elif fault == 'resources':
        (tmp_path / 'tiny_native_resources.json').write_text(json.dumps({'samples': [
            {'mem_available_bytes': 0, 'disk_free_bytes': 0}]}))
    else:
        path = tmp_path / 'progress.json'
        rows = json.loads(path.read_text())
        key, value = {'hash': ('sha256', 'wrong'), 'order': ('case', 'real_native'),
            'validation': ('validation', []), 'manifest': ('manifest_sha256', [])}[fault]
        rows[0][key] = value
        path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        module.audit(tmp_path, allow_partial=True)
