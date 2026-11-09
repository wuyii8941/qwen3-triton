import pytest
import hashlib
import json

from scripts.profile_fsdp_exchange import ORDER, command, helpers, audit_case
from benchmarks.profile_fsdp_exchange_expandable import hashes as worker_hashes
from benchmarks.fsdp_exchange_quality_expandable import ALLOCATOR


@pytest.mark.parametrize('name', ORDER)
def test_explicit_profile_contract(name):
    cmd = command(name, 'unused')
    assert '--profile' in cmd and '--reference-check' in cmd
    assert cmd[cmd.index('--memory-budget-gib') + 1] == '42'
    assert cmd[cmd.index('--keep-full-layers') + 1] == ('32' if name.startswith('decoder') else '36')
    assert ('--exchange-reduce' in cmd) == name.startswith('decoder')
    assert cmd[cmd.index('--warmup') + 1] == '3'
    assert cmd[cmd.index('--steps') + 1] == '12'


def test_unknown_case_rejected():
    with pytest.raises(ValueError):
        command('other', 'unused')


def test_helpers_guard_analysis_and_worker():
    sources = helpers()
    assert 'scripts/analyze_fsdp_trace.py' in sources
    assert 'benchmarks/compare_fsdp_training.py' in sources
    assert all(len(value) == 64 for value in sources.values())


def fixture(root, monkeypatch, decoder=True):
    # Isolate the new evidence layer; base worker validation has separate tests.
    monkeypatch.setattr('scripts.profile_fsdp_exchange.audit_result', lambda *a, **kw: ('matched',))
    counts = {f'model.layers.{i}': 16 for i in range(36)} if decoder else {}
    result = dict(global_parameters=8190735360, world_size=4, batch=1, seq=512, warmup=3, steps=12,
                  profile={}, optimization=dict(exchange_reduce='torch' if decoder else None,
                  exchange_scope='decoder' if decoder else 'all'),
                  communication_environment=dict.fromkeys(('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE')),
                  exchange_callback_counts_by_rank=[dict(counts) for _ in range(4)],
                  manifest=dict(input_sha256_by_rank=['a' * 64] * 4), median_step_seconds=5.)
    raw = json.dumps(result).encode()
    (root / 'result.json').write_bytes(raw)
    trace = json.dumps(dict(traceEvents=[dict(cat='kernel', ph='X', name='ncclKernel',
                                            ts=0, dur=10, args=dict(device=0))])).encode()
    (root / 'trace.json').write_bytes(trace)
    settings = dict(expandable_segments=True, PYTORCH_CUDA_ALLOC_CONF=ALLOCATOR)
    for rank in range(4):
        row = dict(rank=rank, passed=True, sources=worker_hashes(), settings_before=settings,
                   settings_after=settings, effective_cap_gib=41.875,
                   actual_memory_fraction=41.875/48., device_total_bytes=48 * 2**30)
        if rank == 0:
            row.update({'result.json_sha256': hashlib.sha256(raw).hexdigest(),
                        'trace.json_sha256': hashlib.sha256(trace).hexdigest()})
        (root / f'profile_allocator_rank{rank}.json').write_text(json.dumps(row))
    return result


@pytest.mark.parametrize('decoder', [False, True])
def test_audits_all_rank_contexts_and_trace(tmp_path, monkeypatch, decoder):
    fixture(tmp_path, monkeypatch, decoder)
    signature, row = audit_case(tmp_path, 'decoder_a' if decoder else 'native_a')
    assert len(row['allocator_sha256']) == 4
    assert row['trace']['devices']['0']['communication_union_us'] == 10


@pytest.mark.parametrize('field,value', [('rank', 0), ('effective_cap_gib', 42), ('sources', {}),
                                       ('actual_memory_fraction', .5), ('passed', False)])
def test_rejects_corrupt_nonzero_rank(tmp_path, monkeypatch, field, value):
    fixture(tmp_path, monkeypatch)
    path = tmp_path / 'profile_allocator_rank2.json'
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='allocator/source'):
        audit_case(tmp_path, 'decoder_a')


def test_profile_callback_is_not_omitted_from_counts(tmp_path, monkeypatch):
    result = fixture(tmp_path, monkeypatch)
    result['exchange_callback_counts_by_rank'][3]['model.layers.0'] = 15
    (tmp_path / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='callback counts'):
        audit_case(tmp_path, 'decoder_a')


def test_trace_hash_is_bound(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    with (tmp_path / 'trace.json').open('a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError, match='allocator/source'):
        audit_case(tmp_path, 'decoder_a')
