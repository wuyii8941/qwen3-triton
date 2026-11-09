import pytest
import scripts.freeze_fsdp_reference as module


def test_content_identity_is_path_and_mtime_independent(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(b'weights');b.write_bytes(b'weights')
    assert module.snapshot({'weights':a})['content_sha256']==module.snapshot({'weights':b})['content_sha256']
    b.write_bytes(b'changed')
    assert module.snapshot({'weights':a})['content_sha256']!=module.snapshot({'weights':b})['content_sha256']


def test_refuse_changed_file_during_scan(tmp_path,monkeypatch):
    a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(b'a');b.write_bytes(b'b')
    original=module.file_digest
    def change(path):
        value=original(path)
        if path==a:b.write_bytes(b'changed')
        return value
    monkeypatch.setattr(module,'file_digest',change)
    with pytest.raises(ValueError):module.snapshot({'a':a,'b':b})


def test_weight_index_paths_cannot_escape(tmp_path):
    assert module.weight_path(tmp_path,'shard.safetensors')==tmp_path/'shard.safetensors'
    for name in ('../outside.safetensors','/tmp/outside.safetensors','shard.pt'):
        with pytest.raises(ValueError):module.weight_path(tmp_path,name)
    with pytest.raises(ValueError):module.snapshot({})


def test_frozen_manifest_rehash_rejects_mutation_and_tampering(tmp_path):
    a=tmp_path/'weights';a.write_bytes(b'weights')
    report=dict(complete=True,**module.snapshot({'weights':a}))
    assert module.verify_frozen(report)
    a.write_bytes(b'weights')  # mtime alone is not a content change.
    assert module.verify_frozen(report)
    a.write_bytes(b'changed')
    with pytest.raises(ValueError):module.verify_frozen(report)
    a.write_bytes(b'weights');report['files']['weights']['sha256']='0'*64
    with pytest.raises(ValueError):module.verify_frozen(report)
    with pytest.raises(ValueError):module.verify_frozen(dict(complete=False))
