"""Coordinate our research jobs; NVML still controls availability of other owners.

Future GPU research launchers must use this lease while the formal queue is
live. The lock is released during waits for external jobs, and held only for
an acquired run window. External jobs are never stopped or modified.
"""
from contextlib import contextmanager
from pathlib import Path
import fcntl,subprocess,time

root=Path(__file__).resolve().parents[2]


def apps():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
    return [(line.split(',')[0].strip(),int(line.split(',')[1])) for line in raw.splitlines() if line.strip()]


@contextmanager
def lease_when_idle(devices):
    uuid={int(line.split(',')[0]):line.split(',')[1].strip() for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()}
    selected={uuid[i] for i in devices}
    path=root/'.runtime-research/gpu_research.lock';path.parent.mkdir(parents=True,exist_ok=True)
    while True:
        handle=path.open('a+')
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close();time.sleep(5);continue
        try:
            busy=any(gpu in selected for gpu,pid in apps())
        except BaseException:
            fcntl.flock(handle,fcntl.LOCK_UN);handle.close();raise
        if busy:
            fcntl.flock(handle,fcntl.LOCK_UN);handle.close();time.sleep(10);continue
        break
    try:yield selected
    finally:fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
