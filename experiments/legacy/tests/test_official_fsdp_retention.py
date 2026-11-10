"""CPU policy tests; GPU gradient/update parity is a separate gate."""
from types import SimpleNamespace
import pytest
import training.official_fsdp as module


class Unit:
    def __init__(self):self.weight=object()
    def float(self):return self
    def cuda(self):return self


def model():
    body=SimpleNamespace(embed_tokens=Unit(),layers=[Unit(),Unit()],
        norm=Unit(),rotary_emb=Unit())
    return SimpleNamespace(model=body,lm_head=Unit(),config=SimpleNamespace(use_cache=True))


@pytest.mark.parametrize('keep',[0,1,2])
def test_last_layers_only_and_default_unchanged(monkeypatch,keep):
    calls=[]
    monkeypatch.setattr(module,'fully_shard',lambda unit,**kw:calls.append((unit,kw)))
    m=model()
    result=module.shard_official_qwen(m,keep_full_layers=keep)
    assert result is m and not m.config.use_cache
    for i,layer in enumerate(m.model.layers):
        options=next(kw for unit,kw in calls if unit is layer)
        assert options['reshard_after_forward']==(i<2-keep)
    for unit,options in calls:
        if all(unit is not layer for layer in m.model.layers):
            assert options['reshard_after_forward'] is True


@pytest.mark.parametrize('keep',[-1,3])
def test_invalid_retention_rejected_before_mutation(monkeypatch,keep):
    monkeypatch.setattr(module,'fully_shard',lambda *a,**kw:pytest.fail('Must reject before wrapping'))
    with pytest.raises(ValueError):module.shard_official_qwen(model(),keep_full_layers=keep)


def test_snapshot_reshards_each_fsdp_unit(monkeypatch):
    calls=[]
    class Sharded:
        def reshard(self):calls.append(self)
    monkeypatch.setattr(module,'FSDPModule',Sharded)
    root,child=Sharded(),Sharded()
    m=SimpleNamespace(modules=lambda:iter([root,Unit(),child]))
    module.reshard_official_parameters(m)
    assert calls==[root,child]


def test_checkpoint_selection_is_independent_of_retention(monkeypatch):
    import training.fsdp_activation as policy
    wrapped=[];calls=[]
    monkeypatch.setattr(policy,'wrap_decoder_checkpoint',lambda layer,**kw:(wrapped.append((layer,kw)) or layer))
    monkeypatch.setattr(module,'fully_shard',lambda unit,**kw:calls.append((unit,kw)))
    m=model();module.shard_official_qwen(m,keep_full_layers=2,checkpoint_every=2,checkpoint_mode='selective')
    assert wrapped==[(m.model.layers[0],{'mode':'selective'})]
    for layer in m.model.layers:
        assert next(kw for unit,kw in calls if unit is layer)['reshard_after_forward'] is False
