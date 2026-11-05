import json

from benchmarks.watch_training_replication import completed_preflight_checkpoints, prune_preflights


def make_completed(root):
    (root/'preflight_s512_gate.json').write_text(json.dumps(dict(compile={'validation:4':.001},triton={'validation:4':.001})))
    directory=root/'preflight_s512_native';directory.mkdir()
    (directory/'events.jsonl').write_text(json.dumps(dict(event='preflight_interrupted',step=4))+'\n')
    (root/'preflight_s512_native.clock.jsonl').write_text(json.dumps(dict(event='exit',returncode=0))+'\n')
    for name in ('last.pt','best.pt'):
        (directory/name).write_bytes(b'test checkpoint')
    return directory


def test_pruning_never_selects_formal_or_active_checkpoint(tmp_path):
    complete=make_completed(tmp_path)
    formal=tmp_path/'s512_seed20261102_compile';formal.mkdir();(formal/'last.pt').write_bytes(b'keep')
    active=tmp_path/'preflight_s512_compile';active.mkdir();(active/'last.pt').write_bytes(b'keep')
    assert set(completed_preflight_checkpoints(tmp_path))=={complete/'last.pt',complete/'best.pt'}
    removed=prune_preflights(tmp_path)
    assert len(removed)==2
    assert (formal/'last.pt').exists() and (active/'last.pt').exists()
    assert (complete/'events.jsonl').exists()


def test_no_gate_means_no_pruning(tmp_path):
    directory=make_completed(tmp_path)
    (tmp_path/'preflight_s512_gate.json').unlink()
    assert completed_preflight_checkpoints(tmp_path)==[]
    assert (directory/'last.pt').exists()


def test_symlink_checkpoint_is_not_selected(tmp_path):
    directory=make_completed(tmp_path)
    (directory/'last.pt').unlink()
    target=tmp_path/'protected.pt';target.write_bytes(b'keep')
    (directory/'last.pt').symlink_to(target)
    assert directory/'last.pt' not in completed_preflight_checkpoints(tmp_path)
