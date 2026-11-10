"""Isolated bounded-workspace adapter gate; no production-source mutation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.screen_fsdp_1p7b import park_waiting_queue
from training.fsdp_provenance import source_snapshot


def collect_idle(read, *, attempts=20, pause=time.sleep):
    """Bounded stronger guard: three consecutive samples, same thresholds."""
    samples = []
    for attempt in range(attempts):
        readings = read()
        if len(readings) == 4 and all(free >= 8192 and util < 10 for free, util in readings):
            samples.append(readings)
            if len(samples) == 3:
                return samples
        else:
            samples = []
        if attempt + 1 < attempts:
            pause(1)
    return None


def hashes():
    names = ('scripts/verify_fsdp_scratch_bounded.py', 'benchmarks/bf16_scratch_bounded.py',
             'benchmarks/bf16_scratch_validation.py', 'benchmarks/bf16_scratch_pack.py',
             'benchmarks/fsdp_scratch_exchange_adapter.py', 'benchmarks/bf16_exchange_reduce.py',
             'benchmarks/verify_block_fsdp.py')
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}


def worker(output):
    import torch
    from benchmarks.bf16_scratch_bounded import BoundedScratchExchangeReduceScatter as Adapter
    from benchmarks.verify_block_fsdp import main as gate
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    props = torch.cuda.get_device_properties()
    torch.cuda.set_per_process_memory_fraction((4 * 1024**3) / props.total_memory)
    before = hashes()
    sys.argv = ['bounded_scratch_gate', '--backend', 'official', '--bf16',
                '--exchange-reduce', 'triton', '--head-dim', '128', '--seq', '129',
                '--deterministic', '--keep-full-layers', '2', '--checkpoint-every', '1']
    # Explicit test dependency injection only; installed Torch is untouched.
    with patch('benchmarks.fsdp_bf16_exchange_adapter.BF16ExchangeReduceScatter', Adapter):
        gate()
    calls = [instance.calls for instance in Adapter.instances]
    if sum(calls) != 27 or hashes() != before:
        raise ValueError('Missing callback execution or changed sources')
    with (output / f'rank{rank}.json').open('x') as file:
        json.dump(dict(passed=True, rank=rank, callback_calls=calls,
                       actual_adapter=Adapter.__name__, source_sha256=before), file, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--park-pid', type=int)
    parser.add_argument('--park-status', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.output)
        return
    if args.park_pid is None or args.park_status is None:
        raise ValueError('Waiting-only supervisor identity required')
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    args.output.mkdir(parents=True, exist_ok=False)
    before, common = hashes(), source_snapshot(Path.cwd())
    with park_waiting_queue(args.park_pid, args.park_status):
        observations = []
        def read():
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                            '--format=csv,noheader,nounits'], text=True).splitlines()
            resources = [[int(v.strip()) for v in row.split(',')] for row in rows[:4]]
            observations.append(dict(unix_seconds=time.time(), readings=resources))
            return resources
        samples = collect_idle(read)
        (args.output / 'launch_resources.json').write_text(json.dumps(observations, indent=2))
        if samples is None:
            (args.output / 'status.json').write_text(json.dumps(
                dict(phase='resource_guard_rejected', workers_launched=False), indent=2))
            raise RuntimeError('Four idle cards with8GiB free each required')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE'):
            env.pop(key, None)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   '-m', 'scripts.verify_fsdp_scratch_bounded', '--worker', '--output', str(args.output)]
        code = run_owned(command, args.output / 'workers.log', env, timeout=600)
        if code:
            raise RuntimeError(f'Bounded scratch gate failed: {code}')
        ranks = [json.loads((args.output / f'rank{r}.json').read_text()) for r in range(4)]
        if (hashes() != before or source_snapshot(Path.cwd()) != common or any(
                not row['passed'] or row['rank'] != rank or row['source_sha256'] != before
                or row['actual_adapter'] != 'BoundedScratchExchangeReduceScatter'
                or sum(row['callback_calls']) != 27 for rank, row in enumerate(ranks))):
            raise ValueError('Incomplete rank evidence or changed implementation')
        gates = []
        for line in (args.output / 'workers.log').read_text().splitlines():
            if line.startswith('{'):
                try: row = json.loads(line)
                except json.JSONDecodeError: continue
                if row.get('passed') is True and 'tied' in row:
                    gates.append(row)
        if (len(gates) != 2 or {row['tied'] for row in gates} != {True, False}
                or any(row['steps'] != 3 or row['world'] != 4 or row['head_dim'] != 128
                       or row['seq'] != 129 or row['checkpoint_every'] != 1
                       or row['exchange_callback_calls'] != (12 if row['tied'] else 15)
                       for row in gates)):
            raise ValueError('Missing complete tied/untied model gates')
        result = dict(passed=True, actual_adapter='BoundedScratchExchangeReduceScatter',
                      gates=gates, rank_records=ranks, launch_resources=samples[-1],
                      idle_confirmations=3, resource_observations=observations,
                      source_sha256=before, implementation_sources=common,
                      log_sha256=hashlib.sha256((args.output / 'workers.log').read_bytes()).hexdigest(),
                      timing_measured=False, default_changed=False, allocator_cap_gib_per_rank=4,
                      scope='TinyQwen four-card3-update gradient/master gate at existing tolerances; no exact AdamW-state, real8B, performance or convergence claim')
        with (args.output / 'result.json').open('x') as file:
            json.dump(result, file, indent=2)
        print(json.dumps(dict(passed=True, actual_adapter=result['actual_adapter'], updates_each=3)))


if __name__ == '__main__':
    main()
