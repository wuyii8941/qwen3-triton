"""Serial tiny and real post-update quality gates for legacy exchange paths."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from benchmarks.fsdp_exchange_quality import hashes as worker_hashes
from scripts.audit_fsdp_exchange_deterministic import audit as audit_diagnostic
from scripts.audit_fsdp_exchange_confirmation import audit as audit_performance
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.fsdp_resource_budget import budget_ready
from training.fsdp_provenance import source_snapshot


MODES = {'native': (36, None, None, None), 'decoder': (32, 'torch', 'decoder', None),
         'head': (32, 'triton', 'head', 'Simple')}


def command(output, mode, protocol, *, tiny):
    keep, backend, scope, _ = MODES[mode]
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
        '-m', 'benchmarks.fsdp_exchange_quality', '--backend', 'official', '--memory-budget-gib', '42',
        '--keep-full-layers', str(keep), '--checkpoint-every', '1', '--preflight-steps', '2',
        '--protocol', str(protocol), '--output', str(output)]
    if tiny:
        cmd += ['--tiny']
    if backend:
        cmd += ['--exchange-backend', backend, '--exchange-scope', scope]
    return cmd


def audit_job(directory, mode, protocol_path, *, tiny):
    raw = (directory / 'result.json').read_bytes()
    result = json.loads(raw)
    protocol = json.loads(protocol_path.read_text())
    if tiny:
        protocol['sequence_length'] = 16
        protocol['validation']['every_updates'] = 1
    keep, backend, scope, proto = MODES[mode]
    if (not result['complete'] or result['converged'] or result['backend'] != 'official'
            or result['stop_reason'] != 'preflight_cutoff' or 'test' in result
            or result['protocol'] != protocol or not result['best_parameters_restored']
            or result['optimization'] != dict(requested_keep_full_layers=keep,
                keep_full_layers=2 if tiny else keep, checkpoint_every=1, checkpoint_mode='full')
            or result['source_snapshot'] != source_snapshot(Path.cwd())
            or [r['step'] for r in result['training']] != [1, 2]
            or [r['step'] for r in result['validation']] != ([0, 1, 2] if tiny else [0, 2])):
        raise ValueError('Wrong preflight policy/history/protocol or missing best restoration')
    tokens, blocks = (255, 17) if tiny else (261632, 512)
    if any(r['tokens'] != tokens or r['blocks'] != blocks or not math.isfinite(r['loss'])
           for r in result['validation']):
        raise ValueError('Incomplete or nonfinite full validation')
    if any(not math.isfinite(r['loss']) or not 0 <= r['max_rank_peak_allocated_gib'] <= r['max_rank_peak_reserved_gib'] <= 42
           for r in result['training']):
        raise ValueError('Nonfinite training or memory budget violation')
    best = min(result['validation'], key=lambda r: r['loss'])
    if result['best_validation'] != best['loss'] or result['best_step'] != best['step']:
        raise ValueError('Actual minimum validation parameters not selected')
    expected_names = ({f'model.layers.{i}' for i in range(2 if tiny else 36)} if scope == 'decoder'
                      else {'lm_head'} if backend else set())
    env = {k: proto if k == 'NCCL_PROTO' else None for k in
           ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE')}
    manifests = []
    for rank in range(4):
        data = (directory / f'exchange_quality_rank{rank}.json').read_bytes()
        row = json.loads(data)
        if (row['rank'] != rank or row['passed'] is not True or row['source_sha256'] != worker_hashes()
                or row['exchange_backend'] != backend or row['exchange_scope'] != scope
                or row['actual_adapter'] != ('BF16ExchangeReduceScatter' if backend else 'native')
                or row['actual_value_validation_enabled'] is not False or row['memory_budget_gib'] != 42
                or row['attention'] != 'forced Flash SDPA' or row['communication_environment'] != env
                or row['protocol_file_sha256'] != hashlib.sha256(protocol_path.read_bytes()).hexdigest()
                or set(row['callback_counts']) != expected_names or any(v != 2 for v in row['callback_counts'].values())
                or (rank == 0 and row['base_result_sha256'] != hashlib.sha256(raw).hexdigest())):
            raise ValueError('Actual all-rank adapter/configuration not proven')
        manifests.append(hashlib.sha256(data).hexdigest())
    return result, manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def status(phase, **kw):
        (args.output / 'status.json').write_text(json.dumps(dict(phase=phase, **kw), indent=2))
        print(phase, kw, flush=True)
    def interrupt(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    protocol_path = Path('training/fsdp_convergence_protocol_v1.json')
    protocol_bytes = protocol_path.read_bytes()
    original = json.loads(protocol_bytes)
    preflight = copy.deepcopy(original)
    preflight['validation']['every_updates'] = 2
    post_path = args.output / 'post_update_protocol.json'
    post_path.write_text(json.dumps(preflight, indent=2))
    common = source_snapshot(Path.cwd())
    names = ('scripts/preflight_fsdp_exchange_quality.py', 'scripts/audit_fsdp_exchange_deterministic.py',
        'scripts/diagnose_fsdp_exchange.py', 'scripts/audit_fsdp_exchange_confirmation.py',
        'scripts/diagnose_fsdp_1p7b_repeat.py', 'scripts/fsdp_resource_budget.py')
    def hashes():
        return dict(worker_hashes(), **{n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names})
    before = hashes()
    post_bytes = post_path.read_bytes()
    def unchanged():
        if (hashes() != before or source_snapshot(Path.cwd()) != common
                or protocol_path.read_bytes() != protocol_bytes or post_path.read_bytes() != post_bytes):
            raise ValueError('Frozen implementation/protocol changed')
    try:
        status('auditing_prior_gates')
        diagnostic = audit_diagnostic(Path('artifacts/fsdp_exchange_deterministic_v1'))
        performance = audit_performance(Path('artifacts/fsdp_exchange_model_confirmation_v2'))
        if (not all(diagnostic['eligible_for_quality_preflight'].values())
                or not all(r['over_two_percent_both_orders'] for r in performance['decisions'].values())):
            raise ValueError('Both candidates must pass prior numerical and repeated performance gates')
        frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        verify_frozen(frozen)
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
        for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE',
                    'PYTORCH_CUDA_ALLOC_CONF', 'CUBLAS_WORKSPACE_CONFIG'):
            env.pop(key, None)
        records = []
        for tiny in (True, False):
            for mode, (_, _, _, proto) in MODES.items():
                name = ('tiny_' if tiny else 'real_') + mode
                status('waiting_for_idle_gpus', case=name)
                samples = []
                while len(samples) < 3:
                    values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                        '--format=csv,noheader,nounits'], text=True)
                    readings = [tuple(int(v.strip()) for v in line.split(',')) for line in values.splitlines()]
                    ram = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                               if line.startswith('MemAvailable:'))
                    disk = shutil.disk_usage(Path.cwd()).free
                    if budget_ready(readings, allocator_gib=42, free_margin_gib=2.5) and ram >= 160 * 2**30 and disk >= 16 * 2**30:
                        samples.append(dict(unix_seconds=time.time(), readings=readings[:4], mem_available_bytes=ram, disk_free_bytes=disk))
                    else:
                        samples = []
                    if len(samples) < 3:
                        time.sleep(15)
                (args.output / f'{name}_resources.json').write_text(json.dumps(dict(allocator_budget_gib=42.,
                    free_margin_gib=2.5, idle_confirmations=3, samples=samples), indent=2))
                unchanged()
                status('running_preflight', case=name)
                child_env = dict(env)
                if proto:
                    child_env['NCCL_PROTO'] = proto
                selected_protocol = protocol_path if tiny else post_path
                log = args.output / f'{name}.log'
                if run_owned(command(args.output / name, mode, selected_protocol, tiny=tiny), log, child_env, timeout=1800):
                    raise RuntimeError(f'{name} failed; no fallback')
                unchanged()
                result, manifests = audit_job(args.output / name, mode, selected_protocol, tiny=tiny)
                records.append(dict(case=name, source=str(args.output / name / 'result.json'),
                    sha256=hashlib.sha256((args.output / name / 'result.json').read_bytes()).hexdigest(),
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), manifest_sha256=manifests,
                    validation=result['validation'], best_parameters_restored=True))
                (args.output / 'progress.json').write_text(json.dumps(records, indent=2))
        verify_frozen(frozen)
        unchanged()
        (args.output / 'result.json').write_text(json.dumps(dict(complete=True, preflight_only=True,
            full_quality_completed=False, records=records, implementation_sources=common, helper_sources=before,
            original_protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
            frozen_content_sha256=frozen['content_sha256'], defaults_changed=False, checkpoints_written=False), indent=2))
        status('finished')
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
