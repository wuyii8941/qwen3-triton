import json
import signal
import pytest
import scripts.screen_fsdp_1p7b as module


def test_two_card_budget_not_confused_with_full8b():
    assert module.resource_ready([(40000,0),(40000,2)])
    assert not module.resource_ready([(30000,0),(40000,0)])
    assert not module.resource_ready([(40000,10),(40000,0)])
    assert not module.resource_ready([(40000,0)])


def test_workload_explicitly_separate_from_8b(tmp_path):
    cmd=module.worker_command((28,1,'full'),tmp_path,'/data/1p7b')
    assert cmd[cmd.index('--nproc_per_node=2')]=='--nproc_per_node=2'
    assert cmd[cmd.index('--memory-budget-gib')+1]=='24'
    assert cmd[cmd.index('--checkpoint-every')+1]=='1'
    assert '--reference-check' in cmd


def setup_park(monkeypatch,tmp_path,*,busy=False):
    path=tmp_path/'status.json';path.write_text(json.dumps({'phase':'waiting_for_idle_gpus'}))
    calls=[]
    monkeypatch.setattr(module,'identity',lambda pid:('same','T' if calls else 'S'))
    monkeypatch.setattr(module,'has_children',lambda pid:busy)
    monkeypatch.setattr(module.os,'kill',lambda pid,sig:calls.append(sig))
    return path,calls


def test_waiting_queue_resumed_after_success(monkeypatch,tmp_path):
    path,calls=setup_park(monkeypatch,tmp_path)
    with module.park_waiting_queue(123,path):
        assert calls==[signal.SIGSTOP]
    assert calls==[signal.SIGSTOP,signal.SIGCONT]


def test_waiting_queue_resumed_on_failure(monkeypatch,tmp_path):
    path,calls=setup_park(monkeypatch,tmp_path)
    with pytest.raises(RuntimeError):
        with module.park_waiting_queue(123,path):raise RuntimeError('worker failed')
    assert calls==[signal.SIGSTOP,signal.SIGCONT]


def test_never_park_queue_with_live_child(monkeypatch,tmp_path):
    path,calls=setup_park(monkeypatch,tmp_path,busy=True)
    with pytest.raises(ValueError,match='active GPU'):
        with module.park_waiting_queue(123,path):pass
    assert not calls


def test_launch_race_resumes_without_starting_second_job(monkeypatch,tmp_path):
    path,calls=setup_park(monkeypatch,tmp_path)
    checks=iter([False,True])
    monkeypatch.setattr(module,'has_children',lambda pid:next(checks))
    entered=False
    with pytest.raises(ValueError,match='began work'):
        with module.park_waiting_queue(123,path):entered=True
    assert not entered
    assert calls==[signal.SIGSTOP,signal.SIGCONT]
