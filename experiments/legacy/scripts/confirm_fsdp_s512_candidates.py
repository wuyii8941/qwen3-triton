"""Fresh-source S512 ABC/CBA: historical winner versus all-layer recompute.

No test-loss selection, default changes, or automatic convergence claims.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from training.fsdp_provenance import source_snapshot
from scripts.summarize_fsdp_memory import audit_result
from scripts.summarize_fsdp_shape_confirmation import summarize


POLICIES = {'interval_k35': (35, 2, 'full'),
            'all_k32': (32, 1, 'full'), 'all_k36': (36, 1, 'full')}


def command(policy, output, model):
    keep, every, mode = policy
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
            'benchmarks/compare_fsdp_training.py', '--backend', 'official', '--model', model,
            '--keep-full-layers', str(keep), '--checkpoint-every', str(every),
            '--checkpoint-mode', mode, '--memory-budget-gib', '42', '--batch', '1',
            '--seq', '512', '--warmup', '3', '--steps', '12', '--reference-check',
            '--output', str(output)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--preflight-pid', type=int, required=True)
    p.add_argument('--model', default='/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    initial = source_snapshot(Path.cwd())
    env = dict(os.environ)
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.update(PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')

    def status(phase, **fields):
        (a.output/'status.json').write_text(json.dumps(dict(phase=phase, **fields), indent=2))
        print(phase, fields, flush=True)

    def idle(case):
        status('waiting_for_idle_gpus', case=case)
        while True:
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            values = [tuple(int(x.strip()) for x in line.split(',')) for line in rows]
            if len(values) >= 4 and all(m < 2048 and u < 10 for m, u in values[:4]):
                return
            time.sleep(15)

    def run(cmd, name, run_env):
        idle(name)
        if source_snapshot(Path.cwd()) != initial:
            raise ValueError('Implementation changed during campaign')
        status('running', case=name)
        with (a.output/(name+'.log')).open('x') as stream:
            code = subprocess.run(cmd, env=run_env, stdout=stream, stderr=subprocess.STDOUT).returncode
        if code:
            raise RuntimeError(f'{name} failed: {code}; no selecting around failure')

    try:
        status('waiting_for_preflight')
        # Output existence does not prove completion: inspect terminal result fields.
        while True:
            try:
                preflight = json.loads(a.preflight.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                preflight = {}
            if preflight.get('error'):
                raise ValueError('Real preflight failed')
            if preflight.get('complete'):
                break
            try:
                live_command=Path(f'/proc/{a.preflight_pid}/cmdline').read_bytes()
            except FileNotFoundError:
                live_command=b''
            if b'fsdp_convergence_job.py' not in live_command:
                raise ValueError('Preflight launcher exited without a successful terminal result')
            time.sleep(15)
        if (not preflight.get('best_parameters_restored') or preflight.get('synthetic_preflight')
                or preflight.get('stop_reason') != 'preflight_cutoff'
                or preflight.get('source_snapshot') != initial
                or preflight.get('optimization') != dict(requested_keep_full_layers=32,
                    keep_full_layers=32, checkpoint_every=1, checkpoint_mode='full')):
            raise ValueError('Matching real preflight required')
        for every, keep in ((2, 1), (1, 2)):
            name = f'gate_e{every}'
            gate_env = dict(env, CUBLAS_WORKSPACE_CONFIG=':4096:8')
            cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   'benchmarks/verify_block_fsdp.py', '--backend', 'official', '--bf16',
                   '--deterministic', '--head-dim', '128', '--seq', '512',
                   '--keep-full-layers', str(keep), '--checkpoint-every', str(every)]
            run(cmd, name, gate_env)
            rows = [json.loads(line) for line in (a.output/(name+'.log')).read_text().splitlines()
                    if line.startswith('{')]
            expected = dict(passed=True, backend='official', world=4, seq=512, head_dim=128,
                            steps=3, deterministic=True, checkpoint_every=every, keep_full_layers=keep)
            if (len(rows) != 2 or {r.get('tied') for r in rows} != {False, True}
                    or any(any(r.get(k) != v for k, v in expected.items()) for r in rows)):
                raise ValueError('Numerical gate record mismatch')
        names = list(POLICIES)
        spec = None
        for repeat, order in ((0, names), (1, list(reversed(names)))):
            for name in order:
                case = f'r{repeat}_{name}'
                run(command(POLICIES[name], a.output/case, a.model), case, env)
                result = json.loads((a.output/case/'result.json').read_text())
                audit_result(result, POLICIES[name])
                if result['manifest']['implementation_sources'] != initial:
                    raise ValueError('Benchmark source differs from campaign')
                if spec is None:
                    workload = {k: result[k] for k in ('model','world_size','batch','seq','warmup',
                        'steps','memory_budget_gib','precision','attention')}
                    spec = dict(policies=POLICIES, candidate='all_k32', workload=workload)
                    (a.output/'spec.json').write_text(json.dumps(spec, indent=2))
                with (a.output/'status.log').open('a') as stream:
                    stream.write(f'{case} returncode=0\n')
        report = summarize(a.output, spec)
        (a.output/'result.json').write_text(json.dumps(report, indent=2))
        status('finished', scope='Performance only; no convergence or default-policy change')
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
