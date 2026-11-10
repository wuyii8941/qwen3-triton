"""Keep experiment code and historical performance evidence out of runtime."""
import ast
import json
from pathlib import Path
import subprocess
import sys

from scripts.verify_legacy import verify
from training.policy import resolve_policy, validate_exchange
from training.provenance import source_snapshot
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_archived_sources_are_byte_identical():
    assert verify(ROOT / 'experiments/legacy') == 428


def test_runtime_has_no_experimental_dependencies():
    for directory in ('training', 'ops', 'model'):
        for path in (ROOT / directory).rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or '']
                else:
                    continue
                assert not any(n.split('.')[0] in ('experiments', 'benchmarks', 'scripts') for n in names), path


def test_current_backward_has_no_partial_workspace_variants():
    tree = ast.parse((ROOT / 'ops/triton/attention_backward.py').read_text())
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert '_attention_dq_partial_kernel' not in names
    assert '_attention_dk_kernel' not in names
    assert 'attention_backward_triton_fused_kv' not in names
    assert {'_attention_dq_kernel', '_attention_dkdv_fused_kernel', 'attention_backward_triton'} <= names


def test_current_source_snapshot_excludes_archive():
    snapshot = source_snapshot(ROOT)
    assert snapshot == source_snapshot(ROOT)
    assert 'configs/fsdp_8b.json' in snapshot['files']
    assert not any(n.startswith('experiments/') for n in snapshot['files'])


def test_training_entry_help_needs_no_gpu_or_data():
    result = subprocess.run([sys.executable, '-m', 'training.train', '--help'], cwd=ROOT,
                            text=True, capture_output=True, check=True)
    assert '--exchange' in result.stdout
    assert 'official_keep8' not in result.stdout
    assert 'custom_avg' not in result.stdout


def test_policy_has_explicit_retention_not_backend_aliases():
    assert resolve_policy('official', 32, 1, layer_count=36)['keep_full_layers'] == 32
    assert resolve_policy('official', 32, 1, layer_count=2, tiny=True)['keep_full_layers'] == 2
    assert resolve_policy('custom')['keep_full_layers'] == 0
    for backend in ('official_keep8', 'custom_avg', 'unknown'):
        with pytest.raises(ValueError):
            resolve_policy(backend)
    with pytest.raises(ValueError):
        resolve_policy('official', 37, 1, layer_count=36)


def test_exchange_requires_proven_protocol():
    protocol = json.loads((ROOT / 'configs/fsdp_8b.json').read_text())
    validate_exchange('official', 'torch_decoder', protocol)
    for change in ({'global_batch': 8}, {'compute_dtype': 'float32'}):
        with pytest.raises(ValueError):
            validate_exchange('official', 'torch_decoder', dict(protocol, **change))
    with pytest.raises(ValueError):
        validate_exchange('custom', 'torch_decoder', protocol)
