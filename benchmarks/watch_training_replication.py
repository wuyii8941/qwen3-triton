"""Monitor a specific live queue; optionally prune ONLY completed preflights.

Never modifies training code/config, formal checkpoints, or model weights.
Checkpoint files consume disk; unlinking them is not a GPU-memory remedy.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from benchmarks.report_convergence_comparison import events


def completed_preflight_checkpoints(root):
    """An exact six-directory allowlist, not a recursive checkpoint glob."""
    root = root.resolve()
    candidates = []
    for seq in (512, 2048):
        gate = root / f'preflight_s{seq}_gate.json'
        if not gate.is_file() or gate.is_symlink():
            continue
        evidence = json.loads(gate.read_text())
        if set(evidence) != {'compile', 'triton'}:
            continue
        valid = all(
            0 <= value <= (.002 if key.startswith('initial_validation') else .03 if key.startswith('train') else .01)
            for entries in evidence.values() for key, value in entries.items())
        if not valid or any(not e for e in evidence.values()):
            continue
        for backend in ('native', 'compile', 'triton'):
            name = f'preflight_s{seq}_{backend}'
            directory = root / name
            if directory.is_symlink() or directory.resolve().parent != root:
                continue
            rows = events(directory / 'events.jsonl')
            clocks = events(root / f'{name}.clock.jsonl')
            if not rows or not clocks:
                continue
            if (rows[-1].get('event') != 'preflight_interrupted' or rows[-1].get('step') != 4
                    or clocks[-1].get('event') != 'exit' or clocks[-1].get('returncode') != 0):
                continue
            for filename in ('last.pt', 'best.pt'):
                path = directory / filename
                if path.is_file() and not path.is_symlink():
                    candidates.append(path)
    return candidates


def prune_preflights(root):
    paths = completed_preflight_checkpoints(root)
    if not paths:
        return []
    journal = root / 'checkpoint_pruning.jsonl'
    # Journal intent before unlink; individual explicit files, never directories.
    with journal.open('a') as out:
        out.write(json.dumps(dict(event='prune_intent', epoch=time.time(), paths=list(map(str, paths)),
            reason='disk below configured floor; completed preflights only', recoverable=False))+'\n')
        out.flush()
        removed = []
        for path in paths:
            path.unlink()
            removed.append(str(path))
            out.write(json.dumps(dict(event='removed', epoch=time.time(), path=str(path)))+'\n')
            out.flush()
    return removed


def snapshot(root, queue):
    status = dict(epoch=time.time(), queue_pid=queue.pid,
                  disk_free_gib=shutil.disk_usage(root).free/1024**3,
                  host_available_gib=psutil.virtual_memory().available/1024**3)
    try:
        status['queue_alive'] = queue.is_running() and queue.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        status['queue_alive'] = False
    plan = json.loads((root/'plan.json').read_text())
    active = []
    for job in plan['jobs']:
        name = f's{job["seq"]}_seed{job["seed"]}_{job["backend"]}'
        rows = events(root/name/'events.jsonl')
        if rows and rows[-1]['event'] != 'terminal':
            active.append(dict(name=name, latest_event=rows[-1]))
    status['unfinished_jobs_with_events'] = active
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        status['gpu_snapshot_csv'] = result.stdout.strip() if result.returncode==0 else None
    except (OSError, subprocess.TimeoutExpired):
        status['gpu_snapshot_csv'] = None
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--queue-pid', type=int, required=True)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--min-free-gib', type=float, default=100)
    parser.add_argument('--allow-prune-preflights', action='store_true')
    args = parser.parse_args();root=args.root.resolve()
    if args.interval<=0 or args.min_free_gib<=0:
        raise ValueError('Positive interval and disk floor required')
    queue = psutil.Process(args.queue_pid)
    command = queue.cmdline()
    if (not any(module in command for module in ('benchmarks.run_training_replication','benchmarks.recover_remaining_replication')) or '--tag' not in command
            or root.name != 'training_replication_'+command[command.index('--tag')+1]
            or Path(queue.cwd()).resolve() != root.parent.parent):
        raise ValueError('PID does not match the requested queue/root')
    while True:
        status = snapshot(root, queue)
        removed = []
        if status['queue_alive'] and status['disk_free_gib'] < args.min_free_gib and args.allow_prune_preflights:
            removed = prune_preflights(root)
        status.update(removed_checkpoints=removed,
                      disk_warning=status['disk_free_gib'] < args.min_free_gib,
                      policy='only finished/gated preflight last.pt,best.pt; retain all formal checkpoints')
        temp=root/'watchdog.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(root/'watchdog.json')
        with (root/'watchdog_history.jsonl').open('a') as out:
            out.write(json.dumps(status)+'\n')
        print(json.dumps(dict(queue_alive=status['queue_alive'],disk_free_gib=round(status['disk_free_gib'],1),
                              jobs=[x['name'] for x in status['unfinished_jobs_with_events']],removed=len(removed))),flush=True)
        if not status['queue_alive']:
            break
        time.sleep(args.interval)


if __name__=='__main__':
    main()
