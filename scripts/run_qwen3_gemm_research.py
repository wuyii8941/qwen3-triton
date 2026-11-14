"""Run a single-GPU experiment with stable admission and ownership evidence.

Other GPUs may be occupied. Their owners are recorded but are not interference
on this selected device. The shared lock coordinates all our research trees.
"""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, subprocess, time

ROOT = Path(__file__).resolve().parents[1]


def applications():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = []
    for line in raw.splitlines():
        if line.strip():
            gpu, pid, memory = (part.strip() for part in line.split(','))
            rows.append(dict(gpu_uuid=gpu, pid=int(pid), memory_mib=int(memory)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('worker', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.is_relative_to(Path('/home/tzh'))
    assert args.worker and Path(args.worker[0]).is_file()
    output.mkdir(parents=True, exist_ok=False)
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {int(line.split(',')[0]): line.split(',')[1].strip() for line in raw.splitlines()}
    assert args.gpu and len(set(args.gpu)) == len(args.gpu)
    assert all(gpu in uuids for gpu in args.gpu)
    lock = ROOT / '.runtime-research/gpu_research.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    state = dict(stage='waiting_for_120s_idle', candidate_gpus=args.gpu,
                 controller_pid=os.getpid(), command=args.worker,
                 shared_lock=str(lock.resolve()))
    (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    while True:
        handle = lock.open('a+')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close(); time.sleep(5); continue
        admitted, idle = [], {gpu: 0 for gpu in args.gpu}
        selected = None
        while True:
            rows = applications()
            admitted.append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), apps=rows))
            admitted = admitted[-120:]
            busy = {row['gpu_uuid'] for row in rows}
            for gpu in args.gpu:
                idle[gpu] = 0 if uuids[gpu] in busy else idle[gpu] + 1
            if max(idle.values(), default=0) % 10 == 0:
                state.update(idle_observations=idle, latest_observation_utc=admitted[-1]['utc'])
                (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
            ready = [gpu for gpu in args.gpu if idle[gpu] >= 120]
            if ready:
                selected = ready[0]
                break
            if not any(idle.values()):
                break
            time.sleep(1)
        if selected is not None:
            break
        fcntl.flock(handle, fcntl.LOCK_UN); handle.close(); time.sleep(5)
    uuid = uuids[selected]
    assert len(admitted) == 120
    assert all(not any(row['gpu_uuid'] == uuid for row in s['apps']) for s in admitted)
    state.update(gpu=selected, uuid=uuid)
    (output / 'admission.json').write_text(json.dumps(admitted, indent=2) + '\n')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(selected), CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    command = ['bash', str(ROOT / 'scripts/research_python.sh'), *args.worker,
               '--output', str(output / 'measurements')]
    from qwen3_triton.benchmark_utils.monitor import descendants
    samples, owned = [], set()
    try:
        with (output / 'worker.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(stage='running', worker_pid=child.pid)
            (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
            while child.poll() is None:
                owned.update(descendants(child.pid))
                sample = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                try:
                    sample['apps'] = applications()
                    owned.update(descendants(child.pid))
                    sample['foreign_selected_pids'] = [row['pid'] for row in sample['apps']
                        if row['gpu_uuid'] == uuid and row['pid'] not in owned]
                except (ValueError, OSError, subprocess.SubprocessError) as exc:
                    sample['observation_error'] = str(exc)
                samples.append(sample)
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            rc = child.wait()
        clean = not any(s.get('foreign_selected_pids') or s.get('observation_error') for s in samples)
        report = dict(returncode=rc, selected_device_clean=clean, samples=samples,
                      owned_pids=sorted(owned), gpu_uuid=uuid, controller_pid=os.getpid(),
                      launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (output / 'gpu.json').write_text(json.dumps(report, indent=2) + '\n')
        state.update(stage='complete' if rc == 0 and clean else 'failed', returncode=rc,
                     selected_device_clean=clean)
        (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        assert rc == 0 and clean, state
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN); handle.close()


if __name__ == '__main__':
    main()
