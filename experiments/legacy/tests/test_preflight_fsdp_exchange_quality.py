from pathlib import Path
import hashlib
import json

import pytest

from scripts.preflight_fsdp_exchange_quality import command, MODES
from scripts import preflight_fsdp_exchange_quality as module


@pytest.mark.parametrize('mode', list(MODES))
@pytest.mark.parametrize('tiny', [True, False])
def test_preflight_uses_measured_policy_and_explicit_cutoff(mode, tiny):
    cmd = command(Path('unused'), mode, Path('protocol'), tiny=tiny)
    keep, backend, scope, _ = MODES[mode]
    assert cmd[cmd.index('--keep-full-layers') + 1] == str(keep)
    assert cmd[cmd.index('--preflight-steps') + 1] == '2'
    assert cmd[cmd.index('--memory-budget-gib') + 1] == '42'
    assert ('--tiny' in cmd) == tiny
    assert ('--exchange-backend' in cmd) == bool(backend)
    if backend:
        assert cmd[cmd.index('--exchange-backend') + 1] == backend
        assert cmd[cmd.index('--exchange-scope') + 1] == scope


def fixture(tmp_path, monkeypatch, mode, tiny):
    common, sources = {'sha256': 'a' * 64}, {'worker': 'b' * 64}
    monkeypatch.setattr(module, 'source_snapshot', lambda _: common)
    monkeypatch.setattr(module, 'worker_hashes', lambda: sources)
    path = tmp_path / 'protocol.json'
    protocol = {'sequence_length': 512, 'validation': {'every_updates': 2}}
    path.write_text(json.dumps(protocol))
    if tiny:
        protocol = {'sequence_length': 16, 'validation': {'every_updates': 1}}
    keep, backend, scope, proto = MODES[mode]
    tokens, blocks = (255, 17) if tiny else (261632, 512)
    validation = [dict(step=i, loss=3. - i / 10, tokens=tokens, blocks=blocks)
                  for i in ([0, 1, 2] if tiny else [0, 2])]
    result = dict(complete=True, converged=False, backend='official', stop_reason='preflight_cutoff',
        protocol=protocol, best_parameters_restored=True, source_snapshot=common,
        optimization=dict(requested_keep_full_layers=keep, keep_full_layers=2 if tiny else keep,
            checkpoint_every=1, checkpoint_mode='full'),
        training=[dict(step=i, loss=2., max_rank_peak_allocated_gib=39., max_rank_peak_reserved_gib=41.) for i in (1, 2)],
        validation=validation, best_step=2, best_validation=2.8)
    raw = json.dumps(result).encode()
    (tmp_path / 'result.json').write_bytes(raw)
    counts = ({f'model.layers.{i}': 2 for i in range(2 if tiny else 36)} if scope == 'decoder'
              else {'lm_head': 2} if backend else {})
    for rank in range(4):
        row = dict(rank=rank, passed=True, source_sha256=sources, exchange_backend=backend,
            exchange_scope=scope, actual_adapter='BF16ExchangeReduceScatter' if backend else 'native',
            actual_value_validation_enabled=False, memory_budget_gib=42, attention='forced Flash SDPA',
            communication_environment={k: proto if k == 'NCCL_PROTO' else None for k in
                ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE')},
            protocol_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), callback_counts=counts)
        if rank == 0:
            row['base_result_sha256'] = hashlib.sha256(raw).hexdigest()
        (tmp_path / f'exchange_quality_rank{rank}.json').write_text(json.dumps(row))
    return path


@pytest.mark.parametrize('mode', list(MODES))
@pytest.mark.parametrize('tiny', [True, False])
def test_audit_every_preflight_path(tmp_path, monkeypatch, mode, tiny):
    path = fixture(tmp_path, monkeypatch, mode, tiny)
    _, manifests = module.audit_job(tmp_path, mode, path, tiny=tiny)
    assert len(manifests) == 4


@pytest.mark.parametrize('field,value', [('callback_counts', {}), ('actual_value_validation_enabled', True),
    ('source_sha256', {}), ('rank', 0)])
def test_adapter_evidence_cannot_be_mislabeled(tmp_path, monkeypatch, field, value):
    protocol = fixture(tmp_path, monkeypatch, 'decoder', False)
    path = tmp_path / 'exchange_quality_rank2.json'
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        module.audit_job(tmp_path, 'decoder', protocol, tiny=False)
