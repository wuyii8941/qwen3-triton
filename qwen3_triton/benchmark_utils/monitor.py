"""Record live GPU owners during a job without touching CUDA process state."""
import datetime,hashlib,json,os,subprocess,time
from pathlib import Path

def descendants(pid):
    found={pid};pending=[pid]
    while pending:
        current=pending.pop()
        try:children=[int(x) for x in Path(f'/proc/{current}/task/{current}/children').read_text().split()]
        except (OSError,ValueError):continue
        for child in children:
            if child not in found:found.add(child);pending.append(child)
    return found

def run(cmd,*,cwd,env,stdout,monitor):
    from training.provenance import source_snapshot
    runtime_source=source_snapshot(Path(cwd))
    samples=[];owned=set();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with subprocess.Popen(cmd,cwd=cwd,env=env,stdout=stdout,stderr=subprocess.STDOUT) as child:
        while child.poll() is None:
            owned.update(descendants(child.pid));row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
                owned.update(descendants(child.pid));apps=[]
                for line in raw.splitlines():
                    fields=[x.strip() for x in line.split(',')]
                    if len(fields)==3:apps.append(dict(gpu_uuid=fields[0],pid=int(fields[1]),memory_mib=int(fields[2])))
                row['apps']=apps;row['foreign_pids']=[a['pid'] for a in apps if a['pid'] not in owned]
            except (OSError,ValueError,subprocess.SubprocessError) as exc:row['observation_error']=str(exc)
            samples.append(row)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:pass
        rc=child.wait()
    report=dict(command=cmd,cwd=str(cwd),launcher_pid=child.pid,source_snapshot=runtime_source,started_utc=started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=rc,known_owned_pids=sorted(owned),samples=samples,
        foreign_process_seen=any(s.get('foreign_pids') for s in samples),observation_errors=sum('observation_error' in s for s in samples),monitor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(monitor).write_text(json.dumps(report,indent=2)+'\n')
    if rc:raise subprocess.CalledProcessError(rc,cmd)
