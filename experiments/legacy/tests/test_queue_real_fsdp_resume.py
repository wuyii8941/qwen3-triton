import json
import sys
from types import SimpleNamespace
import pytest
from scripts import queue_real_fsdp_resume as queue


@pytest.mark.parametrize('phase',['failed','waiting_for_idle_gpus'])
def test_unsuccessful_predecessor_never_launches(tmp_path,monkeypatch,phase):
    campaign=tmp_path/'campaign';campaign.mkdir()
    (campaign/'status.json').write_text(json.dumps(dict(phase=phase)))
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['queue','--predecessor','-1','--campaign',str(campaign),'--output',str(out)])
    def forbidden(*a,**k):raise AssertionError('Must not launch')
    monkeypatch.setattr(queue.subprocess,'run',forbidden)
    with pytest.raises(ValueError):queue.main()
    assert json.loads((out/'status.json').read_text())['phase']=='failed'


def test_low_disk_guard_does_not_start_four_gpu_replay(tmp_path,monkeypatch):
    campaign=tmp_path/'campaign';campaign.mkdir()
    (campaign/'status.json').write_text('{"phase":"finished"}')
    out=tmp_path/'out';calls=[]
    monkeypatch.setattr(sys,'argv',['queue','--predecessor','-1','--campaign',str(campaign),'--output',str(out)])
    monkeypatch.setattr(queue.subprocess,'check_output',lambda *a,**k:'0,0\n'*4)
    def run(cmd,**kwargs):
        calls.append(cmd)
        assert '--resource-plan' in cmd
        return SimpleNamespace(stdout='{"resources_sufficient":false}')
    monkeypatch.setattr(queue.subprocess,'run',run)
    queue.main()
    assert len(calls)==1
    assert json.loads((out/'status.json').read_text())['phase']=='resource_guard_rejected'
