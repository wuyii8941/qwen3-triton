import json
import pytest
from training.optimization_policy import prepare


def test_default_keeps_original_optimizer(monkeypatch):
    import benchmarks.research_candidates as candidates
    sentinel=object();model=object();calls=[]
    def factory(m,name,lr):calls.append((m,name,lr));return sentinel
    monkeypatch.setattr(candidates,'make_optimizer',factory)
    optimizer,undo=prepare(model,mode='off',optimizer_name='flashoptim',lr=1e-5,seq=512)
    assert optimizer is sentinel and calls==[(model,'flashoptim',1e-5)]
    undo()


@pytest.mark.parametrize('change',[{'passed':False},{'deterministic':False},{'seq':128},{'batch':2}])
def test_reject_wrong_gate(tmp_path,change):
    gate=dict(passed=True,deterministic=True,seq=512,batch=1);gate.update(change)
    path=tmp_path/'gate.json';path.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='gate does not validate'):
        prepare(object(),mode='embedding',optimizer_name='flashoptim',lr=1e-5,seq=512,embedding_gate=path)


def test_require_explicit_gate():
    with pytest.raises(ValueError,match='explicit correctness'):
        prepare(object(),mode='embedding',optimizer_name='flashoptim',lr=1e-5,seq=512)


def test_reject_unvalidated_optimizer():
    with pytest.raises(ValueError,match='only with FlashOptim'):
        prepare(object(),mode='all',optimizer_name='triton',lr=1e-5,seq=512)


def test_consumed_variant_requires_matching_gate(tmp_path):
    path=tmp_path/'gate.json';path.write_text(json.dumps(dict(passed=True,deterministic=True,seq=512,batch=1)))
    with pytest.raises(ValueError,match='own consume_proxy gate'):
        prepare(object(),mode='embedding_consumed',optimizer_name='flashoptim',lr=1e-5,seq=512,embedding_gate=path)


def test_accumulation_requires_matching_gate(tmp_path):
    path=tmp_path/'gate.json';path.write_text(json.dumps(dict(passed=True,deterministic=True,seq=512,batch=2,consume_proxy=True,accumulation=2)))
    with pytest.raises(ValueError,match='accumulation setting'):
        prepare(object(),mode='embedding_consumed',optimizer_name='flashoptim',lr=1e-5,seq=512,
                embedding_gate=path,batch=2,accumulation=4)
