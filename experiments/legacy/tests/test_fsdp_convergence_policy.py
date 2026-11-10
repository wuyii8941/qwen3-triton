import pytest
from training.fsdp_convergence_policy import resolve_policy


def test_explicit_candidate_identity_and_tiny_clamp():
    real=resolve_policy('official',32,1,layer_count=36)
    assert real==dict(requested_keep_full_layers=32,keep_full_layers=32,
                     checkpoint_every=1,checkpoint_mode='full')
    tiny=resolve_policy('official',32,1,layer_count=2,tiny=True)
    assert tiny['keep_full_layers']==2
    assert tiny['requested_keep_full_layers']==32
    with pytest.raises(ValueError):resolve_policy('official',32,1,layer_count=2)


def test_legacy_defaults_unchanged():
    assert resolve_policy('official')['keep_full_layers']==0
    assert resolve_policy('official_keep24',layer_count=2)['keep_full_layers']==2
    assert resolve_policy('official_keep8')['checkpoint_every']==0


@pytest.mark.parametrize('kwargs',[
    dict(backend='custom',checkpoint_every=1),
    dict(backend='custom',keep_full_layers=1),
    dict(backend='official_keep24',keep_full_layers=1),
    dict(backend='official',keep_full_layers=-1),
    dict(backend='official',checkpoint_every=-1),
    dict(backend='official',checkpoint_mode='selective'),
])
def test_invalid_combinations(kwargs):
    with pytest.raises(ValueError):resolve_policy(**kwargs)
