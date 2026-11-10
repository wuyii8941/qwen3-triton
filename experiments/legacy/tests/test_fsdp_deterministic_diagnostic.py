import os
import signal
import sys
import pytest
import scripts.diagnose_fsdp_1p7b_repeat as diagnostic
import benchmarks.deterministic_fsdp_comparison as wrapper


def test_owned_success(tmp_path):
    log=tmp_path/'worker.log'
    assert diagnostic.run_owned([sys.executable,'-c','print("finished")'],log,dict(os.environ))==0
    assert log.read_text().strip()=='finished'


def test_interruption_cleans_owned_launcher_before_propagation(monkeypatch,tmp_path):
    events=[]
    class Process:
        pid=12345
        waits=0
        def poll(self):return None
        def wait(self,timeout):
            self.waits+=1
            if self.waits==1:raise KeyboardInterrupt()
            events.append('reaped');return 0
    monkeypatch.setattr(diagnostic.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(diagnostic.os,'killpg',lambda pid,sig:events.append((pid,sig)))
    with pytest.raises(KeyboardInterrupt):
        diagnostic.run_owned(['owned'],tmp_path/'worker.log',{})
    assert events==[(12345,signal.SIGTERM),'reaped']


def test_wrapper_requires_explicit_workspace_before_running_model(monkeypatch,tmp_path):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    monkeypatch.setattr(sys,'argv',['wrapper','--output',str(tmp_path)])
    with pytest.raises(ValueError,match='workspace'):wrapper.main()


def test_validation_timeout_is_explicit_not_a_change_in_numerical_gate(monkeypatch,tmp_path):
    received=[]
    class Process:
        def wait(self,timeout):received.append(timeout);return 0
        def poll(self):return 0
    monkeypatch.setattr(diagnostic.subprocess,'Popen',lambda *a,**k:Process())
    assert diagnostic.run_owned(['owned'],tmp_path/'worker.log',{},timeout=1800)==0
    assert received==[1800]
