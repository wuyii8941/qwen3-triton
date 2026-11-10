import json
import pytest
from scripts.confirm_fsdp_shape import select_policies,verify_gate


def row(name,seconds,every,status='passed'):
    return dict(case=name,median_step_seconds=seconds,status=status,
        policy=dict(keep_full_layers=16,checkpoint_every=every,checkpoint_mode='full'))


def test_select_strongest_successful_native_not_oom_or_default():
    rows=[row('native16',7,0),row('native18',6.8,0),row('native19',0,0,'oom'),
          row('full28',6,2),row('full29',6.2,2)]
    policies,candidate=select_policies(dict(records=rows))
    assert list(policies)==['native18','full28'] and candidate=='full28'
    with pytest.raises(ValueError):select_policies(dict(records=rows[:2]))


@pytest.mark.parametrize('fault',[None,'missing','seq','deterministic','policy','passed'])
def test_gate_requires_both_matching_records(tmp_path,fault):
    rows=[dict(passed=True,tied=tied,backend='official',dtype='torch.bfloat16',world=4,
        head_dim=128,seq=1024,steps=3,deterministic=True,checkpoint_every=2,
        checkpoint_mode='full',keep_full_layers=1,loss_chunk_size=0,swiglu=False) for tied in (False,True)]
    if fault=='missing':rows.pop()
    elif fault=='seq':rows[0]['seq']=11
    elif fault=='deterministic':rows[0]['deterministic']=False
    elif fault=='policy':rows[0]['checkpoint_every']=0
    elif fault=='passed':rows[0]['passed']=False
    path=tmp_path/'gate.log';path.write_text('\n'.join(json.dumps(r) for r in rows))
    if fault:
        with pytest.raises(ValueError):verify_gate(path,2,1024)
    else:verify_gate(path,2,1024)
