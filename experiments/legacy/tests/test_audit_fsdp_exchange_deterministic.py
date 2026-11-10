import hashlib
import json

import pytest

from scripts import audit_fsdp_exchange_deterministic as module


def fixture(root, monkeypatch, *, complete):
    monkeypatch.chdir(root)
    common, helpers = {'sha256': 'a' * 64}, {'worker': 'b' * 64}
    monkeypatch.setattr(module, 'source_snapshot', lambda _: common)
    monkeypatch.setattr(module, 'hashes', lambda: helpers)
    monkeypatch.setattr(module, 'audit_launch', lambda *a, **kw: 'c' * 64)
    contexts = ['d' * 64] * 4
    monkeypatch.setattr(module, 'audit_short', lambda directory, name:
        (json.loads((directory / 'result.json').read_text()), ('same',), contexts))
    rows, runs = [], {}
    names = list(module.CASES) if complete else list(module.CASES)[:1]
    for name in names:
        path = root / name
        path.mkdir()
        run = {'records': [{'loss': x} for x in (2.65, 2.68, 2.73, 2.30)]}
        raw = json.dumps(run).encode()
        (path / 'result.json').write_bytes(raw)
        log = b'synthetic, not GPU evidence'
        (root / f'{name}.log').write_bytes(log)
        (root / f'{name}_resources.json').write_text(json.dumps({'samples': [
            {'mem_available_bytes': 160 * 2**30, 'disk_free_bytes': 16 * 2**30}]}))
        rows.append(dict(case=name, source=str(path / 'result.json'),
            sha256=hashlib.sha256(raw).hexdigest(), log_sha256=hashlib.sha256(log).hexdigest(),
            context_sha256=contexts))
        runs[name] = run
    (root / 'progress.json').write_text(json.dumps(rows))
    if complete:
        (root / 'results').mkdir()
        (root / 'results/frozen_fsdp_reference_v1.json').write_text(json.dumps({'content_sha256': 'e' * 64}))
        (root / 'status.json').write_text(json.dumps({'phase': 'finished'}))
        (root / 'result.json').write_text(json.dumps(dict(complete=True, records=rows,
            checks=module.conclusions(runs), helper_sources=helpers, implementation_sources=common,
            defaults_changed=False, losses={n: [r['loss'] for r in run['records']] for n, run in runs.items()},
            frozen_content_sha256='e' * 64)))


def test_partial_has_no_eligibility(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch, complete=False)
    with pytest.raises(ValueError):
        module.audit(tmp_path)
    result = module.audit(tmp_path, allow_partial=True)
    assert not result['complete'] and result['raw_records_audited'] == 1
    assert result['checks'] is None and not result['eligible_for_quality_preflight']


def test_complete_recomputes_loss_checks(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch, complete=True)
    result = module.audit(tmp_path)
    assert result['complete'] and result['raw_records_audited'] == 6
    assert all(result['eligible_for_quality_preflight'].values())
    assert not result['convergence_proven'] and not result['full_state_equality_proven']


@pytest.mark.parametrize('fault', ['raw', 'log', 'context', 'order', 'decision', 'resource'])
def test_corrupt_evidence_rejected(tmp_path, monkeypatch, fault):
    fixture(tmp_path, monkeypatch, complete=True)
    if fault == 'raw':
        path = tmp_path / 'auto_a/result.json'
        path.write_bytes(path.read_bytes() + b' ')
    elif fault == 'log':
        (tmp_path / 'auto_a.log').write_text('changed')
    elif fault == 'resource':
        path = tmp_path / 'auto_a_resources.json'
        path.write_text(json.dumps({'samples': [{'mem_available_bytes': 0, 'disk_free_bytes': 0}]}))
    else:
        path = tmp_path / 'result.json'
        value = json.loads(path.read_text())
        if fault == 'context':
            value['records'][0]['context_sha256'] = []
        elif fault == 'order':
            value['records'].reverse()
        else:
            value['checks']['decoder']['eligible_for_quality_preflight'] = False
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.audit(tmp_path)
