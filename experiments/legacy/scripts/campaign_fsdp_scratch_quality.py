"""Gated serial quality campaign. Safety cutoff never counts as convergence."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from benchmarks.fsdp_scratch_bounded_quality import hashes as quality_sources
from scripts.audit_fsdp_convergence import audit as audit_history
from scripts.audit_fsdp_scratch_bounded import audit as audit_performance
from scripts.audit_fsdp_scratch_deterministic import audit as audit_diagnostic
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.fsdp_resource_budget import budget_ready
from scripts.screen_fsdp_1p7b import park_waiting_queue
from training.flashoptim_convergence_job import Blocks
from training.fsdp_provenance import source_snapshot


def select_keep(report):
    if not report['complete'] or not report['decisions']['bounded_decoder_k34']['over_two_percent_both_orders']:
        raise ValueError('Repeated decoder performance gate required')
    rows = {row['case']: row for row in report['records']}
    feasible = [k for k in (32, 34) if all(rows[f'r{r}_native_k{k}']['status'] == 'passed' for r in (0, 1))]
    if not feasible:
        raise ValueError('No native control passed both orders')
    choice = min(feasible, key=lambda k: sum(rows[f'r{r}_native_k{k}']['median_step_seconds'] for r in (0, 1)))
    return choice


def command(output, *, keep, bounded, protocol, tiny=False, preflight=False):
    if tiny and not preflight:
        raise ValueError('Tiny must have a preflight cutoff')
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
           '-m', 'benchmarks.fsdp_scratch_bounded_quality', '--backend', 'official',
           '--memory-budget-gib', '40', '--keep-full-layers', str(keep), '--checkpoint-every', '1',
           '--protocol', str(protocol), '--output', str(output)]
    if bounded:
        cmd += ['--bounded-exchange-scope', 'decoder']
    if tiny:
        cmd += ['--tiny']
    if preflight:
        cmd += ['--preflight-steps', '2']
    return cmd


def audit_job(directory, *, keep, bounded, protocol, protocol_file_sha256, tiny=False, preflight=False):
    raw = (directory / 'result.json').read_bytes()
    result = json.loads(raw)
    expected = copy.deepcopy(protocol)
    if tiny:
        expected['sequence_length'] = 16
        expected['validation']['every_updates'] = 1
    policy = result.get('optimization', {})
    if (not result['complete'] or result['backend'] != 'official' or result['protocol'] != expected
            or not result['best_parameters_restored']
            or policy.get('requested_keep_full_layers') != keep
            or policy.get('keep_full_layers') != (2 if tiny else keep)
            or policy.get('checkpoint_every') != 1 or policy.get('checkpoint_mode') != 'full'):
        raise ValueError('Wrong complete policy/protocol/best-restore job')
    if preflight:
        steps = [0, 1, 2] if tiny else [0, 2]
        tokens, blocks = (255, 17) if tiny else (261632, 512)
        if (result['converged'] or result['stop_reason'] != 'preflight_cutoff' or 'test' in result
                or [v['step'] for v in result['training']] != [1, 2]
                or [v['step'] for v in result['validation']] != steps
                or any(v['tokens'] != tokens or v['blocks'] != blocks for v in result['validation'])):
            raise ValueError('Missing full post-update validation or invalid preflight claim')
        history = dict(preflight_only=True, post_update_validation_complete=True)
    else:
        history = audit_history(result, expected, 512, 583)
    if any(not math.isfinite(v['loss']) or v['max_rank_peak_allocated_gib'] > 40
           or v['max_rank_peak_reserved_gib'] > 40 for v in result['training']):
        raise ValueError('Nonfinite loss or budget violation')
    manifests, hashes = [], quality_sources()
    updates = len(result['training'])
    expected_units = {f'model.layers.{i}' for i in range(2 if tiny else 36)} if bounded else set()
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    for rank in range(4):
        path = directory / f'quality_adapter_rank{rank}.json'
        data = path.read_bytes()
        row = json.loads(data)
        if (row['rank'] != rank or not row['passed'] or row['source_sha256'] != hashes
                or row['bounded_scope'] != ('decoder' if bounded else None)
                or row['actual_adapter'] != ('BoundedScratchExchangeReduceScatter' if bounded else 'native')
                or row['actual_value_validation_enabled'] != bounded or row['memory_budget_gib'] != 40
                or row['attention'] != 'forced Flash SDPA'
                or row['protocol_file_sha256'] != protocol_file_sha256
                or set(row['actual_callback_counts']) != expected_units
                or any(v != updates for v in row['actual_callback_counts'].values())):
            raise ValueError('Actual quality adapter/settings/callback count not proven on every rank')
        if rank == 0 and row['base_result_sha256'] != hashlib.sha256(raw).hexdigest():
            raise ValueError('Quality manifest does not bind raw job')
        manifests.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest()))
    return result, dict(history=history, adapter_manifests=manifests,
                        protocol_content_sha256=protocol_hash, pretrained_parity_from_prior_gate=not tiny)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--park-pid', type=int, required=True)
    parser.add_argument('--park-status', type=Path, required=True)
    parser.add_argument('--launch-long', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def status(phase, **fields):
        (args.output / 'status.json').write_text(json.dumps(dict(phase=phase, **fields), indent=2))
        print(phase, fields, flush=True)
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    def hashes():
        names = ('scripts/campaign_fsdp_scratch_quality.py', 'scripts/audit_fsdp_scratch_bounded.py',
                 'scripts/audit_fsdp_scratch_deterministic.py', 'scripts/diagnose_fsdp_scratch_bounded.py',
                 'scripts/audit_fsdp_convergence.py', 'scripts/diagnose_fsdp_1p7b_repeat.py',
                 'scripts/fsdp_resource_budget.py', 'scripts/screen_fsdp_1p7b.py', 'scripts/freeze_fsdp_reference.py')
        return dict(quality_sources(), **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names})
    before, common = hashes(), source_snapshot(Path.cwd())
    protocol_path = Path('training/fsdp_convergence_protocol_v1.json')
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    preflight_protocol = copy.deepcopy(protocol)
    preflight_protocol['validation']['every_updates'] = 2
    preflight_path = args.output / 'post_update_protocol.json'
    preflight_path.write_text(json.dumps(preflight_protocol, indent=2))
    preflight_bytes = preflight_path.read_bytes()
    frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
    def unchanged():
        if (hashes() != before or source_snapshot(Path.cwd()) != common
                or protocol_path.read_bytes() != protocol_bytes or preflight_path.read_bytes() != preflight_bytes):
            raise ValueError('Quality implementation or frozen protocol changed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE',
                'CUBLAS_WORKSPACE_CONFIG', 'PYTORCH_CUDA_ALLOC_CONF'):
        env.pop(key, None)
    def idle(label):
        status('waiting_for_idle_gpus', case=label, allocator_budget_gib=40, free_margin_gib=2.5)
        samples = []
        while True:
            values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                                              '--format=csv,noheader,nounits'], text=True).splitlines()
            readings = [[int(v.strip()) for v in row.split(',')] for row in values[:4]]
            ram = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                           if line.startswith('MemAvailable:'))) * 1024
            disk = shutil.disk_usage(Path.cwd()).free
            ready = budget_ready(readings, allocator_gib=40., free_margin_gib=2.5) and ram >= 160 * 2**30 and disk >= 16 * 2**30
            samples = samples + [dict(unix_seconds=time.time(), readings=readings,
                                     mem_available_bytes=ram, disk_free_bytes=disk)] if ready else []
            if len(samples) == 3:
                (args.output / (label + '_resources.json')).write_text(json.dumps(dict(
                    allocator_budget_gib=40, free_margin_gib=2.5, idle_confirmations=3, samples=samples,
                    min_ram_available_gib=160, min_disk_free_gib=16), indent=2))
                return
            time.sleep(15)
    try:
        status('verifying_performance_diagnostic_and_inputs')
        performance = audit_performance(Path('artifacts/fsdp_scratch_bounded_budget40_v1'))
        diagnostic = audit_diagnostic(Path('artifacts/fsdp_scratch_bounded_deterministic_v1'))
        if not diagnostic['eligible_for_quality_preflight']:
            raise ValueError('Deterministic numerical gate not passed')
        for first in (Path('artifacts/fsdp_scratch_bounded_budget40_v1/r0_native_k32/result.json'),
                      Path('artifacts/fsdp_scratch_bounded_deterministic_v1/native_a/result.json')):
            if json.loads(first.read_text())['manifest']['implementation_sources'] != common:
                raise ValueError('Performance/diagnostic common source differs from quality source')
        keep = select_keep(performance)
        (args.output / 'selection.json').write_text(json.dumps(dict(native_keep=keep, bounded_keep=34,
            bounded_scope='decoder', selection_uses_test_loss=False, source='fresh matched performance only'), indent=2))
        verify_frozen(frozen); unchanged()
        with park_waiting_queue(args.park_pid, args.park_status):
            rows, completed = [], {}
            stages = [('tiny_native', keep, False, True, True, protocol_path, protocol),
                      ('tiny_bounded', 34, True, True, True, protocol_path, protocol),
                      ('real_native_preflight', keep, False, False, True, preflight_path, preflight_protocol),
                      ('real_bounded_preflight', 34, True, False, True, preflight_path, preflight_protocol)]
            if args.launch_long:
                stages += [('native_full_protocol', keep, False, False, False, protocol_path, protocol),
                           ('bounded_full_protocol', 34, True, False, False, protocol_path, protocol)]
            for name, policy, bounded, tiny, preflight, config_path, config in stages:
                idle(name); unchanged(); status('running_quality_job', case=name, preflight=preflight)
                path, log = args.output / name, args.output / (name + '.log')
                timeout = 1800 if preflight else protocol['safety']['max_hours_per_backend'] * 3600 + 1800
                if run_owned(command(path, keep=policy, bounded=bounded, protocol=config_path,
                                     tiny=tiny, preflight=preflight), log, env, timeout=timeout):
                    raise RuntimeError(f'{name}: job failed; no fallback or weaker protocol')
                unchanged()
                result, evidence = audit_job(path, keep=policy, bounded=bounded, protocol=config,
                                             protocol_file_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                                             tiny=tiny, preflight=preflight)
                if result['source_snapshot'] != common:
                    raise ValueError('Performance/quality common implementation mismatch')
                completed[name] = result
                rows.append(dict(case=name, source=str(path / 'result.json'),
                    sha256=hashlib.sha256((path / 'result.json').read_bytes()).hexdigest(),
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), evidence=evidence))
                (args.output / 'progress.json').write_text(json.dumps(rows, indent=2))
            comparison = None
            if args.launch_long:
                a, b = completed['native_full_protocol'], completed['bounded_full_protocol']
                comparison = dict(native_seconds=a['elapsed_seconds'], bounded_seconds=b['elapsed_seconds'],
                    bounded_over_native_job_time=b['elapsed_seconds'] / a['elapsed_seconds'],
                    native_updates=len(a['training']), bounded_updates=len(b['training']),
                    native_median_recorded_step_seconds=statistics.median(v['seconds'] for v in a['training']),
                    bounded_median_recorded_step_seconds=statistics.median(v['seconds'] for v in b['training']),
                    test_loss_difference=b['test']['loss'] - a['test']['loss'],
                    scope='Single-seed shared-hardware complete task-plateau observation; not statistical quality equivalence or isolated causal implementation speedup')
            verify_frozen(frozen); unchanged()
            with (args.output / 'result.json').open('x') as file:
                json.dump(dict(complete=True, full_quality_completed=args.launch_long, records=rows,
                    full_job_comparison=comparison, helper_sources=before, implementation_sources=common,
                    frozen_content_sha256=frozen['content_sha256'], protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
                    default_changed=False, checkpoints_written=False, best_parameters_exported=False), file, indent=2)
        status('finished' if args.launch_long else 'preflights_passed', original_42gib_supervisor_resumed=True)
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
