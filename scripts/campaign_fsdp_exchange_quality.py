"""Full original-protocol native K36 versus separate-buffer decoder K32."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from benchmarks.fsdp_exchange_quality import hashes as worker_hashes
from scripts.audit_fsdp_convergence import audit as audit_history
from scripts.audit_fsdp_exchange_confirmation import audit as audit_performance, audit_launch
from scripts.audit_fsdp_exchange_deterministic import audit as audit_diagnostic
from scripts.audit_fsdp_exchange_quality_preflights import audit as audit_preflights
from scripts.audit_fsdp_scratch_quality import full_comparison
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.fsdp_resource_budget import budget_ready
from scripts.preflight_fsdp_exchange_quality import MODES, command as preflight_command
from training.fsdp_provenance import source_snapshot


def command(output, mode, protocol):
    if mode not in ('native', 'decoder'):
        raise ValueError('Only the preselected undominated decoder and native control')
    cmd = preflight_command(output, mode, protocol, tiny=False)
    i = cmd.index('--preflight-steps')
    del cmd[i:i + 2]
    return cmd


def selection(rows):
    indexed = {(r['repeat'], r['case']): r for r in rows}
    for repeat in (0, 1):
        decoder = indexed[(repeat, 'torch_decoder_auto_k32')]
        head = indexed[(repeat, 'triton_head_simple_k32')]
        if (decoder['status'] != 'passed' or head['status'] != 'passed'
                or decoder['median_step_seconds'] >= head['median_step_seconds']
                or decoder['peak_allocated_gib'] >= head['peak_allocated_gib']):
            raise ValueError('Measured dominance required; do not choose from validation/test')
    return dict(native_keep=36, candidate_keep=32, candidate_backend='torch', candidate_scope='decoder',
        attention='forced Flash SDPA', memory_budget_gib=42., selection_uses_validation_or_test_loss=False,
        head_pruned_reason='Decoder is faster and lower allocated memory in both measured orders')


def audit_full(directory, mode, protocol_path):
    raw = (directory / 'result.json').read_bytes()
    result = json.loads(raw)
    protocol = json.loads(protocol_path.read_text())
    keep, backend, scope, _ = MODES[mode]
    if (result['backend'] != 'official' or result['source_snapshot'] != source_snapshot(Path.cwd())
            or result['optimization'] != dict(requested_keep_full_layers=keep, keep_full_layers=keep,
                checkpoint_every=1, checkpoint_mode='full')):
        raise ValueError('Wrong full quality implementation/policy')
    history = audit_history(result, protocol, 512, 583)
    if any(not math.isfinite(r['loss']) or not 0 <= r['max_rank_peak_allocated_gib'] <= r['max_rank_peak_reserved_gib'] <= 42
           for r in result['training']):
        raise ValueError('Nonfinite loss or full quality budget violation')
    expected = {f'model.layers.{i}' for i in range(36)} if backend else set()
    env = {k: None for k in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE')}
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
                or set(row['callback_counts']) != expected
                or any(v != len(result['training']) for v in row['callback_counts'].values())
                or (rank == 0 and row['base_result_sha256'] != hashlib.sha256(raw).hexdigest())):
            raise ValueError('All-rank full quality adapter proof failed')
        manifests.append(hashlib.sha256(data).hexdigest())
    return result, dict(history=history, manifest_sha256=manifests)


def helpers():
    names = ('scripts/campaign_fsdp_exchange_quality.py', 'scripts/audit_fsdp_exchange_quality.py',
        'scripts/preflight_fsdp_exchange_quality.py', 'scripts/audit_fsdp_exchange_quality_preflights.py',
        'scripts/audit_fsdp_exchange_confirmation.py', 'scripts/audit_fsdp_exchange_deterministic.py',
        'scripts/diagnose_fsdp_exchange.py', 'scripts/diagnose_fsdp_1p7b_repeat.py',
        'scripts/audit_fsdp_convergence.py', 'scripts/audit_fsdp_scratch_quality.py',
        'scripts/fsdp_resource_budget.py', 'scripts/freeze_fsdp_reference.py')
    return dict(worker_hashes(), **{n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names})


def audit_campaign(root, *, allow_partial=False):
    final = root / 'result.json'
    complete = final.exists()
    if not complete and not allow_partial:
        raise ValueError('Complete native/candidate plateau jobs required')
    summary = json.loads(final.read_text()) if complete else None
    rows = summary['records'] if complete else json.loads((root / 'progress.json').read_text())
    if ([r['case'] for r in rows] != ['native', 'decoder'][:len(rows)] or len(rows) > 2
            or (complete and len(rows) != 2)):
        raise ValueError('Wrong ordered full job coverage')
    protocol = Path('training/fsdp_convergence_protocol_v1.json')
    performance = json.loads(Path('artifacts/fsdp_exchange_model_confirmation_v2/result.json').read_text())
    chosen = json.loads((root / 'selection.json').read_text())
    if chosen != selection(performance['records']):
        raise ValueError('Wrong non-quality-based selection')
    runs, evidence = {}, []
    for row in rows:
        name = row['case']
        path = root / name / 'result.json'
        if (Path(row['source']).resolve() != path.resolve()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']
                or hashlib.sha256((root / f'{name}.log').read_bytes()).hexdigest() != row['log_sha256']):
            raise ValueError('Full job raw result/log changed')
        result, audited = audit_full(root / name, name, protocol)
        if audited != row['evidence']:
            raise ValueError('History/adapter evidence mismatch')
        resource_sha = audit_launch(root, name,
            dict(allocator_budget_gib=42., free_margin_gib=2.5, idle_confirmations=3))
        resources = json.loads((root / f'{name}_resources.json').read_text())
        if any(s['mem_available_bytes'] < 160 * 2**30 or s['disk_free_bytes'] < 16 * 2**30 for s in resources['samples']):
            raise ValueError('RAM/disk resource guard violated')
        runs[name] = result
        evidence.append(dict(case=name, result_sha256=row['sha256'], resource_sha256=resource_sha, **audited))
    comparison = full_comparison(runs['native'], runs['decoder']) if complete else None
    if complete:
        if (summary['complete'] is not True or summary['full_quality_completed'] is not True
                or summary['implementation_sources'] != source_snapshot(Path.cwd())
                or summary['helper_sources'] != helpers() or summary['comparison'] != comparison
                or summary['protocol_sha256'] != hashlib.sha256(protocol.read_bytes()).hexdigest()
                or summary['defaults_changed'] is not False or summary['checkpoints_written'] is not False
                or summary['best_parameters_exported'] is not False
                or json.loads((root / 'status.json').read_text())['phase'] != 'finished'):
            raise ValueError('Full campaign summary/source/status mismatch')
        frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        if summary['frozen_content_sha256'] != frozen['content_sha256']:
            raise ValueError('Frozen input identity mismatch')
    return dict(complete=complete, partial_diagnostic_only=not complete, raw_jobs_audited=len(rows),
        evidence=evidence, comparison=comparison, full_quality_completed=complete,
        defaults_changed=False, checkpoints_written=False, best_parameters_exported=False,
        statistical_quality_equivalence_proven=False)


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
    common, before = source_snapshot(Path.cwd()), helpers()
    protocol = Path('training/fsdp_convergence_protocol_v1.json')
    protocol_bytes = protocol.read_bytes()
    def unchanged():
        if source_snapshot(Path.cwd()) != common or helpers() != before or protocol.read_bytes() != protocol_bytes:
            raise ValueError('Full quality source/protocol changed')
    try:
        status('auditing_prior_gates')
        preflights = audit_preflights(Path('artifacts/fsdp_exchange_quality_preflights_v1'))
        diagnostic = audit_diagnostic(Path('artifacts/fsdp_exchange_deterministic_v1'))
        performance_root = Path('artifacts/fsdp_exchange_model_confirmation_v2')
        performance = audit_performance(performance_root)
        raw_performance = json.loads((performance_root / 'result.json').read_text())
        if (not preflights['all_preflights_passed'] or not diagnostic['eligible_for_quality_preflight']['decoder']
                or not performance['decisions']['torch_decoder_auto_k32']['over_two_percent_both_orders']
                or raw_performance['implementation_sources'] != common):
            raise ValueError('All matched-source prior gates required')
        chosen = selection(raw_performance['records'])
        (args.output / 'selection.json').write_text(json.dumps(chosen, indent=2))
        frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        verify_frozen(frozen)
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
        for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE', 'CUBLAS_WORKSPACE_CONFIG', 'PYTORCH_CUDA_ALLOC_CONF'):
            env.pop(key, None)
        rows, runs = [], {}
        for mode in ('native', 'decoder'):
            status('waiting_for_idle_gpus', case=mode)
            samples = []
            while len(samples) < 3:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True)
                readings = [tuple(int(v.strip()) for v in line.split(',')) for line in raw.splitlines()]
                ram = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
                disk = shutil.disk_usage(Path.cwd()).free
                if budget_ready(readings, allocator_gib=42, free_margin_gib=2.5) and ram >= 160 * 2**30 and disk >= 16 * 2**30:
                    samples.append(dict(unix_seconds=time.time(), readings=readings[:4], mem_available_bytes=ram, disk_free_bytes=disk))
                else:
                    samples = []
                if len(samples) < 3:
                    time.sleep(15)
            (args.output / f'{mode}_resources.json').write_text(json.dumps(dict(allocator_budget_gib=42.,
                free_margin_gib=2.5, idle_confirmations=3, samples=samples), indent=2))
            unchanged()
            status('running_full_quality', case=mode)
            log = args.output / f'{mode}.log'
            timeout = json.loads(protocol_bytes)['safety']['max_hours_per_backend'] * 3600 + 1800
            if run_owned(command(args.output / mode, mode, protocol), log, env, timeout=timeout):
                raise RuntimeError(f'{mode} failed; no fallback')
            unchanged()
            result, evidence = audit_full(args.output / mode, mode, protocol)
            runs[mode] = result
            rows.append(dict(case=mode, source=str(args.output / mode / 'result.json'),
                sha256=hashlib.sha256((args.output / mode / 'result.json').read_bytes()).hexdigest(),
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), evidence=evidence))
            (args.output / 'progress.json').write_text(json.dumps(rows, indent=2))
        verify_frozen(frozen)
        unchanged()
        (args.output / 'result.json').write_text(json.dumps(dict(complete=True, full_quality_completed=True,
            records=rows, comparison=full_comparison(runs['native'], runs['decoder']),
            implementation_sources=common, helper_sources=before,
            protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(), frozen_content_sha256=frozen['content_sha256'],
            defaults_changed=False, checkpoints_written=False, best_parameters_exported=False), indent=2))
        status('finished')
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
