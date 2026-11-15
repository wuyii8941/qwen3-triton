"""Wait for the actual availability probe, then measure only supported peer RS."""
import argparse,json,os,time
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from run_vocab_resource_queue_v12 import run_attempt,dump
from training.provenance import source_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();probe=a.probe.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    state=json.loads((probe/'state.json').read_text());pid=state['pid']
    proc=Path(f'/proc/{pid}/stat');start=proc.read_text().rsplit(') ',1)[1].split()[19] if proc.exists() else None
    while not (probe/'probe.json').exists():
        fields=proc.read_text().rsplit(') ',1)[1].split() if proc.exists() else None
        if fields is None or fields[0] in ['Z','X'] or fields[19]!=start:raise RuntimeError('Availability probe ended without a result')
        time.sleep(5)
    # The result is written before the monitor receipt. Wait for terminal
    # controller state, verifying that the same process is still live.
    while json.loads((probe/'state.json').read_text())['stage'] not in ['complete_availability_probe','resource_contaminated_no_result_adopted']:
        fields=proc.read_text().rsplit(') ',1)[1].split() if proc.exists() else None
        if fields is None or fields[0] in ['Z','X']:raise RuntimeError('Probe ended before clean monitor review')
        time.sleep(2)
    check=json.loads((probe/'probe.json').read_text());monitor=json.loads((probe/'monitor.json').read_text())
    assert check['complete'] and monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
    out.mkdir()
    if not check['available']:
        dump(out/'result.json',dict(complete=True,available=False,full_training_qualified=False,reason='CUDA symmetric-memory probe unavailable',probe_sha256=sha(probe/'probe.json')))
        return
    files=[Path(__file__),root/'scripts/symmetric_rs_v13.py',root/'scripts/benchmark_symmetric_rs_v13.py',root/'scripts/research_python.sh',root/'scripts/run_vocab_resource_queue_v12.py',probe/'probe.json',probe/'monitor.json']
    sources={str(p):sha(p) for p in files};core=source_snapshot(root)
    dump(out/'manifest.json',dict(sources=sources,core_snapshot=core,full_training_qualified=False,policy='first clean supported attempt; numerical/technical failures stop; no timing based retries'))
    attempt=0
    while True:
        dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease'))
        with stable_lease([0,1,2,3]):
            for path,digest in sources.items():assert sha(path)==digest,path
            assert source_snapshot(root)==core
            attempt+=1;folder=out/f'attempt_{attempt:03d}';folder.mkdir()
            command=['bash',str(root/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(root/'scripts/benchmark_symmetric_rs_v13.py'),'--output',str(folder/'run')]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            env.pop('NCCL_PROTO',None);env.pop('NCCL_P2P_LEVEL',None)
            dump(out/'state.json',dict(pid=os.getpid(),stage='running',attempt=attempt))
            monitor=run_attempt(command,env,folder,core)
        if monitor['foreign_process_seen'] or monitor['observation_errors']:continue
        assert monitor['returncode']==0,folder
        result=json.loads((folder/'run/result.json').read_text());assert result['complete']
        for path,digest in sources.items():assert sha(path)==digest,path
        assert source_snapshot(root)==core
        (out/'accepted').symlink_to(folder.name,target_is_directory=True)
        dump(out/'state.json',dict(pid=os.getpid(),stage='complete_module_only'))
        break


if __name__=='__main__':main()
