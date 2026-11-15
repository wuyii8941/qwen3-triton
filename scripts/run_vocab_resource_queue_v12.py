"""Keep fixed role order and retry only resource-contaminated GPU attempts.

First complete clean attempt wins; timing and numerical outcomes never choose
an attempt. Foreign activity or observation failure cancels only our torchrun
agent. Clean technical failures stop the queue. Frozen model math is unchanged.
"""
from pathlib import Path
import argparse,datetime,json,os,subprocess,time
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.gpu_lease import apps
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.monitor import descendants
from training.provenance import source_snapshot
from run_vocab_screen_v12 import ROOT,command


def dump(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def run_attempt(cmd,env,folder,core):
    samples=[];owned=set();reason=None
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (folder/'worker.log').open('w') as log:
        with subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as child:
            launcher=child.pid
            while child.poll() is None:
                owned.update(descendants(child.pid));row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                try:
                    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
                    owned.update(descendants(child.pid));observed=[]
                    for line in raw.splitlines():
                        fields=[s.strip() for s in line.split(',')]
                        if len(fields)==3:observed.append(dict(gpu_uuid=fields[0],pid=int(fields[1]),memory_mib=int(fields[2])))
                    row.update(apps=observed,foreign_pids=[p['pid'] for p in observed if p['pid'] not in owned])
                    if row['foreign_pids']:reason='foreign_GPU_process'
                except (OSError,ValueError,subprocess.SubprocessError) as exc:
                    row['observation_error']=str(exc);reason='GPU_observation_failure'
                samples.append(row)
                if reason:
                    # Popen's exact live child is our agent, never the foreign PID.
                    if child.poll() is None:child.terminate()
                    try:child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait()
                    break
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:pass
            rc=child.wait()
    report=dict(command=cmd,cwd=str(ROOT),launcher_pid=launcher,known_owned_pids=sorted(owned),
        source_snapshot=core,started_utc=started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        returncode=rc,samples=samples,foreign_process_seen=any(s.get('foreign_pids') for s in samples),
        observation_errors=sum('observation_error' in s for s in samples),cancelled_for_resource=reason,
        monitor_implementation=str(Path(__file__).resolve()),monitor_sha256=sha(Path(__file__)))
    dump(folder/'monitor.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prototype',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();prototype=args.prototype.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    manifest=json.loads(prototype.read_text())
    assert manifest['source_root']==str(ROOT) and len(manifest['roles'])==10
    for path,digest in manifest['sources'].items():assert sha(path)==digest,path
    assert source_snapshot(ROOT)==manifest['core_snapshot']
    out.mkdir();(out/'attempts').mkdir()
    manifest['sources'].update({str(Path(__file__).resolve()):sha(Path(__file__)),
        str(ROOT/'scripts/review_vocab_resources_v12.py'):sha(ROOT/'scripts/review_vocab_resources_v12.py'),str(prototype):sha(prototype)})
    manifest['resource_policy']='Fixed role order. For each role accept its first returncode0 complete27-update attempt with all GPU observations owned/error-free. Exclude and cancel any resource-contaminated attempt, retain every log/monitor/hash, and require a new continuous120s all4idle admission. Never choose by timing or numerical pass flags. Clean technical failures stop. No external process is signalled.'
    dump(out/'manifest.json',manifest);history=[];dump(out/'attempts.json',history)
    def verify():
        for path,digest in manifest['sources'].items():assert sha(path)==digest,path
        assert source_snapshot(ROOT)==manifest['core_snapshot']
    def state(stage,**extra):
        a=dict(pid=os.getpid(),stage=stage,**extra);dump(out/'state.json',a);print(json.dumps(a),flush=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    for role in manifest['roles']:
        while True:
            verify();state('waiting_stable_four_GPU_lease',role=role['name'],accepted_roles=sum(r['accepted'] for r in history))
            admitted=False
            try:
                with stable_lease([0,1,2,3]):
                    verify();assert not apps()
                    admitted=True
                    number=len(history)+1;folder=out/'attempts'/f'{number:04d}_{role["name"]}';folder.mkdir()
                    state('running_'+role['name'],attempt=number,accepted_roles=sum(r['accepted'] for r in history))
                    cmd=command(role,folder/'run',full=True)
                    cmd[cmd.index('--checkpoint-every')+1]=str(role['checkpoint_every'])
                    monitor=run_attempt(cmd,env,folder,manifest['core_snapshot']);verify()
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                if admitted:raise
                # Admission errors have no benchmark result; retain the observation.
                state('admission_observation_retry',role=role['name'],error=str(exc));time.sleep(5);continue
            dirty=monitor['foreign_process_seen'] or monitor['observation_errors']
            row=dict(number=number,role=role['name'],folder=str(folder.relative_to(out)),accepted=False,
                     returncode=monitor['returncode'],resource_contaminated=bool(dirty),
                     monitor_sha256=sha(folder/'monitor.json'),log_sha256=sha(folder/'worker.log'))
            if dirty:
                history.append(row);dump(out/'attempts.json',history)
                state('rejected_resource_attempt',role=role['name'],attempt=number);continue
            if monitor['returncode']:
                history.append(row);dump(out/'attempts.json',history)
                state('clean_technical_failure',role=role['name'],attempt=number)
                raise RuntimeError('Clean failed attempt is not automatically retried; inspect its preserved log')
            result=json.loads((folder/'run/result.json').read_text())
            assert result['complete'] and not result['screen_only'] and len(result['ranks'])==4
            assert all(r['updates']==27 and r['parameter_count']==r['optimizer_state_count']==399 for r in result['ranks'])
            row.update(accepted=True,result_sha256=sha(folder/'run/result.json'),trace_sha256=sha(folder/'run/training_trace.json'))
            history.append(row);dump(out/'attempts.json',history)
            (out/role['name']).symlink_to((folder/'run').relative_to(out),target_is_directory=True)
            (out/(role['name']+'.gpu.json')).symlink_to((folder/'monitor.json').relative_to(out))
            (out/(role['name']+'.log')).symlink_to((folder/'worker.log').relative_to(out))
            state('accepted_clean_role',role=role['name'],attempt=number,accepted_roles=sum(r['accepted'] for r in history))
            break
    verify();state('GPU_complete_independent_CPU_review')
    cpu=env.copy();cpu['CUDA_VISIBLE_DEVICES']=''
    subprocess.run(['bash',str(ROOT/'scripts/research_python.sh'),str(ROOT/'scripts/review_vocab_resources_v12.py'),
                    '--output',str(out)],cwd=ROOT,env=cpu,check=True)
    a=json.loads((out/'independent_review.json').read_text());assert a['complete'] and a['audit_passed']
    verify();state('complete_reviewed_no_default_change',qualified_training_improvement=a['qualified_training_improvement'])


if __name__=='__main__':main()
