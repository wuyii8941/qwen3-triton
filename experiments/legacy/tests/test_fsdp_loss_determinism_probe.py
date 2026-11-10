import json
import pytest
from scripts.probe_fsdp_loss_determinism import classify


def test_gradient_negative_is_distinct_from_crash():
    assert classify(1,'AssertionError: Parameter embedding: Tensor-likes are not close!')=='gradient_parity_failed'
    for text in ('CUDA out of memory','SyntaxError','TimeoutError',''):
        with pytest.raises(ValueError):classify(1,text)


def test_success_requires_complete_gate():
    rows=[dict(passed=True,tied=t,deterministic=True,loss_chunk_size=7,seq=11,head_dim=128,steps=3)
          for t in (False,True)]
    assert classify(0,'\n'.join(json.dumps(r) for r in rows))=='passed'
    rows[0]['seq']=129
    with pytest.raises(ValueError):classify(0,'\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):classify(0,'')
