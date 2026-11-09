"""Exercise launcher policy gates without starting subprocesses or CUDA."""
import json
import sys
from pathlib import Path

import pytest
from scripts import run_fsdp_convergence as launcher


def tiny_record():
    return dict(backend='official',complete=True,converged=False,
        stop_reason='preflight_cutoff',best_parameters_restored=True,
        validation=[dict(tokens=255)]*3,
        optimization=dict(requested_keep_full_layers=32,keep_full_layers=2,
                          checkpoint_every=1,checkpoint_mode='full'))


@pytest.mark.parametrize('mutation', ['missing','keep','interval','mode','backend'])
def test_mismatched_tiny_never_launches_real_job(tmp_path,monkeypatch,mutation):
    record=tiny_record()
    if mutation=='missing':record.pop('optimization')
    elif mutation=='backend':record['backend']='official_keep24'
    else:
        key={'keep':'requested_keep_full_layers','interval':'checkpoint_every','mode':'checkpoint_mode'}[mutation]
        record['optimization'][key]={'keep':24,'interval':2,'mode':'selective'}[mutation]
    tiny=tmp_path/'tiny.json';tiny.write_text(json.dumps(record))
    out=tmp_path/'campaign'
    monkeypatch.setattr(sys,'argv',['launcher','--output',str(out),'--tiny-result',str(tiny),
        '--keep-full-layers','32','--checkpoint-every','1','--launch-long'])
    def forbidden(*args,**kwargs):raise AssertionError('GPU subprocess must not launch')
    monkeypatch.setattr(launcher.subprocess,'run',forbidden)
    with pytest.raises(ValueError):launcher.main()
    assert json.loads((out/'status.json').read_text())['phase']=='failed'


@pytest.mark.parametrize('fault',[None,'missing_post_update_validation','wrong_protocol'])
def test_identical_policy_forwarded_to_real_preflight_and_long(tmp_path,monkeypatch,fault):
    tiny=tmp_path/'tiny.json';tiny.write_text(json.dumps(tiny_record()))
    out=tmp_path/'campaign';calls=[]
    monkeypatch.setattr(sys,'argv',['launcher','--output',str(out),'--tiny-result',str(tiny),
        '--keep-full-layers','32','--checkpoint-every','1','--launch-long'])
    monkeypatch.setattr(launcher.subprocess,'check_output',lambda *a,**k:'0, 0\n'*4)
    def run(cmd,**kwargs):
        calls.append(cmd)
        target=Path(cmd[cmd.index('--output')+1]);target.mkdir()
        # Mock completion is solely for testing command forwarding, not quality proof.
        record=dict(complete=True,converged=False,
            best_parameters_restored=True,stop_reason='preflight_cutoff')
        if '--protocol' in cmd:
            record.update(protocol=json.loads(Path(cmd[cmd.index('--protocol')+1]).read_text()),
                          validation=[dict(step=0),dict(step=2)])
            if fault=='missing_post_update_validation':record['validation']=[dict(step=0)]
            if fault=='wrong_protocol':record['protocol']['validation']['every_updates']=32
        (target/'result.json').write_text(json.dumps(record))
    monkeypatch.setattr(launcher.subprocess,'run',run)
    if fault:
        with pytest.raises(ValueError):launcher.main()
        assert len(calls)==1
        assert json.loads((out/'status.json').read_text())['phase']=='failed'
        return
    launcher.main()
    assert len(calls)==2
    for cmd in calls:
        assert cmd[cmd.index('--keep-full-layers')+1]=='32'
        assert cmd[cmd.index('--checkpoint-every')+1]=='1'
        assert cmd[cmd.index('--checkpoint-mode')+1]=='full'
    assert '--preflight-steps' in calls[0] and '--preflight-steps' not in calls[1]
    assert '--protocol' in calls[0] and '--protocol' not in calls[1]
