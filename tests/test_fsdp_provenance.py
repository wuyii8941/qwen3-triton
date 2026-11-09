import pytest
from training.fsdp_provenance import source_snapshot


def make_project(tmp_path):
    for directory in ('training', 'ops', 'model', 'benchmarks', 'artifacts'):
        (tmp_path/directory).mkdir()
    (tmp_path/'benchmarks/compare_fsdp_training.py').write_text('# worker\n')
    (tmp_path/'training/loss.py').write_text('# version 1\n')
    return tmp_path


def test_hash_tracks_dependencies_but_excludes_outputs(tmp_path):
    root = make_project(tmp_path)
    first = source_snapshot(root)
    assert first == source_snapshot(root)
    (root/'artifacts/result.json').write_text('{}')
    (root/'training/checkpoint.pt').write_bytes(b'not source')
    assert first == source_snapshot(root)
    (root/'training/loss.py').write_text('# version 2\n')
    assert source_snapshot(root)['sha256'] != first['sha256']
    (root/'ops/new_kernel.py').write_text('# kernel\n')
    assert 'ops/new_kernel.py' in source_snapshot(root)['files']


def test_rejects_missing_worker_and_external_symlink(tmp_path):
    with pytest.raises(FileNotFoundError): source_snapshot(tmp_path)
    root = tmp_path/'project'; root.mkdir(); make_project(root)
    outside = tmp_path/'external.py'; outside.write_text('# external\n')
    (root/'ops/link.py').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'): source_snapshot(root)
