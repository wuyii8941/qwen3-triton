"""Serial four-rank native/head/decoder smoke for the real-step wrapper."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmarks.compare_fsdp_scratch_bounded import sources
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.screen_fsdp_1p7b import park_waiting_queue
from scripts.verify_fsdp_scratch_bounded import collect_idle
from training.fsdp_provenance import source_snapshot


def command(scope, output):
    if scope not in (None, 'head', 'decoder'):
        raise ValueError('Unsupported smoke scope')
    # Tiny exercises plumbing only. The base worker deliberately reserves
    # --reference-check for pretrained checkpoints, not random tiny configs.
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
           '-m', 'benchmarks.compare_fsdp_scratch_bounded', '--backend', 'official',
           '--tiny', '--seq', '129', '--batch', '1', '--warmup', '1', '--steps', '2',
           '--keep-full-layers', '2', '--checkpoint-every', '1', '--memory-budget-gib', '8',
           '--output', str(output)]
    if scope:
        cmd += ['--bounded-exchange-scope', scope]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--park-pid', type=int, required=True)
    parser.add_argument('--park-status', type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    args.output.mkdir(parents=True, exist_ok=False)
    before, common = sources(), source_snapshot(Path.cwd())
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE',
                'CUBLAS_WORKSPACE_CONFIG', 'PYTORCH_CUDA_ALLOC_CONF'):
        env.pop(key, None)
    with park_waiting_queue(args.park_pid, args.park_status):
        observations = []
        def read():
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                            '--format=csv,noheader,nounits'], text=True).splitlines()
            values = [[int(v.strip()) for v in row.split(',')] for row in rows[:4]]
            observations.append(dict(unix_seconds=time.time(), readings=values))
            return values
        idle = collect_idle(read)
        (args.output / 'launch_resources.json').write_text(json.dumps(observations, indent=2))
        if idle is None:
            (args.output / 'status.json').write_text(json.dumps(dict(phase='resource_guard_rejected')))
            raise RuntimeError('Four cards did not become idle')
        records = []
        signatures = set()
        for scope in (None, 'head', 'decoder'):
            label = scope or 'native'
            output, log = args.output / label, args.output / (label + '.log')
            code = run_owned(command(scope, output), log, env, timeout=600)
            if code:
                raise RuntimeError(f'{label} smoke failed: {code}')
            path = output / 'result.json'
            raw = path.read_bytes()
            base = json.loads(raw)
            if (not base['complete'] or base['global_parameters'] >= 1000000
                    or base['world_size'] != 4 or base['warmup'] != 1 or base['steps'] != 2
                    or len(base['records']) != 3 or base['memory_budget_gib'] != 8
                    or 'initial_hf_parity' in base):
                raise ValueError('Wrong complete tiny training workload')
            signatures.add(tuple(base['manifest']['input_sha256_by_rank']))
            manifests = [json.loads((output / f'bounded_manifest_rank{r}.json').read_text()) for r in range(4)]
            for rank, row in enumerate(manifests):
                expected = 'BoundedScratchExchangeReduceScatter' if scope else 'native'
                if (not row['passed'] or row['rank'] != rank or row['bounded_scope'] != scope
                        or row['actual_adapter'] != expected or row['source_sha256'] != before
                        or row['actual_value_validation_enabled'] != bool(scope)):
                    raise ValueError('Actual adapter identity/validation not proven')
                expected_calls = list(base['exchange_callback_counts_by_rank'][rank].values())
                if row['actual_callback_counts'] != expected_calls or any(v != 3 for v in expected_calls):
                    raise ValueError('Actual callback counts differ')
            if manifests[0]['base_result_sha256'] != hashlib.sha256(raw).hexdigest():
                raise ValueError('Wrapper does not bind raw base result')
            records.append(dict(case=label, passed=True, source=str(path),
                                sha256=hashlib.sha256(raw).hexdigest(),
                                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                                actual_manifests=manifests))
        if sources() != before or source_snapshot(Path.cwd()) != common or len(signatures) != 1:
            raise ValueError('Sources or ordered rank inputs differ across smoke paths')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != own_hash:
            raise ValueError('Smoke supervisor source changed')
        with (args.output / 'result.json').open('x') as file:
            json.dump(dict(passed=True, records=records, source_sha256=before,
                           supervisor_sha256=own_hash, implementation_sources=common,
                           performance_claim=False, hf_reference_check_run=False,
                           scope='Tiny D16 wrapper plumbing smoke only; not parity, real8B, convergence or performance'), file, indent=2)
        print(json.dumps(dict(passed=True, cases=[r['case'] for r in records])), flush=True)


if __name__ == '__main__':
    main()
