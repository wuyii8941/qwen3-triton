"""Matched-retention legacy exchange numerical diagnostic, never timing evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from benchmarks.deterministic_fsdp_exchange import hashes
from scripts.audit_fsdp_exchange_confirmation import audit as audit_performance, loss_comparison
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.fsdp_resource_budget import budget_ready
from scripts.summarize_fsdp_memory import audit_result
from training.fsdp_provenance import source_snapshot


CASES = {'auto_a': (None, 'all', None), 'auto_b': (None, 'all', None),
         'simple_a': (None, 'all', 'Simple'), 'simple_b': (None, 'all', 'Simple'),
         'decoder': ('torch', 'decoder', None), 'head': ('triton', 'head', 'Simple')}


def command(name, output):
    backend, scope, _ = CASES[name]
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
           '-m', 'benchmarks.deterministic_fsdp_exchange', '--backend', 'official',
           '--model', '/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
           '--seq', '512', '--batch', '1', '--warmup', '1', '--steps', '3',
           '--keep-full-layers', '32', '--checkpoint-every', '1', '--memory-budget-gib', '42',
           '--reference-check', '--output', str(output)]
    if backend:
        cmd += ['--exchange-reduce', backend, '--exchange-scope', scope]
    return cmd


def audit_short(directory, name):
    raw = (directory / 'result.json').read_bytes()
    result = json.loads(raw)
    signature = audit_result(result, (32, 1, 'full'), allow_exchange=True)
    backend, scope, protocol = CASES[name]
    expected_env = {k: protocol if k == 'NCCL_PROTO' else None for k in
                    ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE')}
    if (result['global_parameters'] != 8190735360 or result['world_size'] != 4
            or result['seq'] != 512 or result['batch'] != 1 or result['warmup'] != 1
            or result['steps'] != 3 or result['memory_budget_gib'] != 42
            or result['optimization']['exchange_reduce'] != backend
            or result['optimization']['exchange_scope'] != scope
            or result['communication_environment'] != expected_env):
        raise ValueError('Wrong matched-retention diagnostic workload')
    diagnostic = result['deterministic_diagnostic']
    if (diagnostic['enabled'] is not True or diagnostic['cublas_workspace'] != ':4096:8'
            or diagnostic['source_hashes'] != {Path(n).name: h for n, h in hashes().items()
                if n in ('benchmarks/deterministic_fsdp_comparison.py', 'benchmarks/compare_fsdp_training.py')}
            or result['manifest']['implementation_sources'] != source_snapshot(Path.cwd())):
        raise ValueError('Missing deterministic worker evidence')
    inputs = result['manifest']['input_sha256_by_rank']
    if len(inputs) != 4 or any(len(s) != 64 for s in inputs):
        raise ValueError('All-rank ordered input identities required')
    expected = {f'model.layers.{i}' for i in range(36)} if scope == 'decoder' else {'lm_head'} if backend else set()
    contexts = []
    for rank in range(4):
        counts = result['exchange_callback_counts_by_rank'][rank]
        path = directory / f'exchange_deterministic_rank{rank}.json'
        context_raw = path.read_bytes()
        context = json.loads(context_raw)
        if (set(counts) != expected or any(v != 4 for v in counts.values())
                or context['rank'] != rank or context['algorithms_enabled'] is not True
                or context['warn_only'] is not False or context['cublas_workspace'] != ':4096:8'
                or context['source_sha256'] != hashes() or context['communication_environment'] != expected_env
                or (rank == 0 and context['result_sha256'] != hashlib.sha256(raw).hexdigest())):
            raise ValueError('All-rank settings/callback/source proof failed')
        contexts.append(hashlib.sha256(context_raw).hexdigest())
    return result, signature + (tuple(inputs),), contexts


def conclusions(runs):
    if set(runs) != set(CASES) or any(len(r['records']) != 4 for r in runs.values()):
        raise ValueError('All six complete four-update diagnostics required')
    reports = {}
    for candidate, first, second in (('decoder', 'auto_a', 'auto_b'), ('head', 'simple_a', 'simple_b')):
        exact = all(a['loss'] == b['loss'] for a, b in zip(runs[first]['records'], runs[second]['records']))
        checks = [loss_comparison(runs[candidate], runs[n]) for n in (first, second)]
        reports[candidate] = dict(native_repeat_exact=exact, comparisons=checks,
            eligible_for_quality_preflight=exact and all(c['within_declared_loss_tolerance'] for c in checks))
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def status(phase, **fields):
        (args.output / 'status.json').write_text(json.dumps(dict(phase=phase, **fields), indent=2))
        print(phase, fields, flush=True)
    def interrupt(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    common = source_snapshot(Path.cwd())
    before = hashes()
    frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
    def unchanged():
        if hashes() != before or source_snapshot(Path.cwd()) != common:
            raise ValueError('Source changed during diagnostic')
    try:
        status('auditing_completed_performance')
        performance_root = Path('artifacts/fsdp_exchange_model_confirmation_v2')
        performance = audit_performance(performance_root)
        summary = json.loads((performance_root / 'result.json').read_text())
        if not performance['complete'] or summary['implementation_sources'] != common:
            raise ValueError('Completed matching-source performance campaign required')
        verify_frozen(frozen)
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
        for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE', 'PYTORCH_CUDA_ALLOC_CONF'):
            env.pop(key, None)
        runs, records, signatures = {}, [], []
        for name, (_, _, protocol) in CASES.items():
            status('waiting_for_idle_gpus', case=name)
            samples = []
            while True:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True)
                readings = [tuple(int(v.strip()) for v in line.split(',')) for line in raw.splitlines()]
                available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemAvailable:'))
                disk = shutil.disk_usage(Path.cwd()).free
                if budget_ready(readings, allocator_gib=42, free_margin_gib=2.5) and available >= 160 * 2**30 and disk >= 16 * 2**30:
                    samples.append(dict(unix_seconds=time.time(), readings=readings[:4],
                        mem_available_bytes=available, disk_free_bytes=disk))
                else:
                    samples = []
                if len(samples) == 3:
                    break
                time.sleep(15)
            (args.output / f'{name}_resources.json').write_text(json.dumps(dict(
                allocator_budget_gib=42., free_margin_gib=2.5, idle_confirmations=3, samples=samples), indent=2))
            unchanged()
            status('running_diagnostic', case=name)
            child_env = dict(env)
            if protocol:
                child_env['NCCL_PROTO'] = protocol
            log = args.output / f'{name}.log'
            if run_owned(command(name, args.output / name), log, child_env, timeout=1800):
                raise RuntimeError(f'{name} failed; no fallback or tolerance change')
            unchanged()
            run, signature, contexts = audit_short(args.output / name, name)
            runs[name] = run
            signatures.append(signature)
            records.append(dict(case=name, source=str(args.output / name / 'result.json'),
                sha256=hashlib.sha256((args.output / name / 'result.json').read_bytes()).hexdigest(),
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), context_sha256=contexts))
            (args.output / 'progress.json').write_text(json.dumps(records, indent=2))
        if any(s != signatures[0] for s in signatures):
            raise ValueError('Ordered inputs or workload identity differ')
        verify_frozen(frozen)
        unchanged()
        result = dict(complete=True, records=records, checks=conclusions(runs),
            losses={n: [r['loss'] for r in run['records']] for n, run in runs.items()},
            implementation_sources=common, helper_sources=before,
            frozen_content_sha256=frozen['content_sha256'], defaults_changed=False,
            scope='Four-update matched-K32 loss-only deterministic diagnostic; not full gradient/moment equality, ordinary performance or convergence')
        (args.output / 'result.json').write_text(json.dumps(result, indent=2))
        status('finished')
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
