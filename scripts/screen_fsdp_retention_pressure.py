"""ABCD/DCBA test of lower retention versus allocator retry/synchronization."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys

from benchmarks.screen_fsdp_retention_expandable import hashes as worker_hashes
from benchmarks.fsdp_exchange_quality_expandable import ALLOCATOR, EFFECTIVE_CAP_GIB, validate_settings
from scripts.audit_fsdp_profiles import audit as audit_profiles, step_summary
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.preflight_fsdp_exchange_expandable import environment, resources
from scripts.summarize_fsdp_memory import audit_result
from training.fsdp_provenance import source_snapshot


CASES = {'native_k36': (36, False), 'native_k24': (24, False),
         'decoder_k32': (32, True), 'decoder_k24': (24, True)}


def command(name, output):
    keep, decoder = CASES[name]
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
           '--module', 'benchmarks.screen_fsdp_retention_expandable', '--backend', 'official',
           '--seq', '512', '--batch', '1', '--warmup', '3', '--steps', '12',
           '--keep-full-layers', str(keep), '--checkpoint-every', '1',
           '--memory-budget-gib', str(EFFECTIVE_CAP_GIB), '--reference-check', '--output', str(output)]
    if decoder:
        cmd += ['--exchange-reduce', 'torch', '--exchange-scope', 'decoder']
    return cmd


def helpers():
    names = ('scripts/screen_fsdp_retention_pressure.py', 'scripts/audit_fsdp_profiles.py',
             'scripts/profile_fsdp_exchange.py', 'scripts/analyze_fsdp_trace.py',
             'scripts/preflight_fsdp_exchange_expandable.py', 'scripts/diagnose_fsdp_1p7b_repeat.py',
             'scripts/fsdp_resource_budget.py', 'scripts/summarize_fsdp_memory.py')
    return dict(worker_hashes(), **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names})


def audit_run(root, name):
    keep, decoder = CASES[name]
    raw = (root / 'result.json').read_bytes()
    result = json.loads(raw)
    signature = audit_result(result, (keep, 1, 'full'), allow_exchange=True)
    if (result['global_parameters'] != 8190735360 or result['world_size'] != 4
            or result['batch'] != 1 or result['seq'] != 512 or result['warmup'] != 3
            or result['steps'] != 12 or 'profile' in result or result['memory_budget_gib'] != EFFECTIVE_CAP_GIB
            or result['optimization']['exchange_reduce'] != ('torch' if decoder else None)
            or result['optimization']['exchange_scope'] != ('decoder' if decoder else 'all')):
        raise ValueError('Wrong retention screen workload')
    if result['communication_environment'] != dict.fromkeys(
            ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE')):
        raise ValueError('Unmatched NCCL configuration')
    counts = result['exchange_callback_counts_by_rank']
    if len(counts) != 4 or any((len(row) != 36 or any(value != 15 for value in row.values()))
                             if decoder else bool(row) for row in counts):
        raise ValueError('Actual callback coverage mismatch')
    contexts = []
    for rank in range(4):
        context_raw = (root / f'retention_allocator_rank{rank}.json').read_bytes()
        row = json.loads(context_raw)
        for key in ('settings_before', 'settings_after'):
            validate_settings(row[key], 'native', ALLOCATOR, None)
        if (row['rank'] != rank or row['passed'] is not True or row['sources'] != worker_hashes()
                or row['settings_before'] != row['settings_after'] or row['effective_cap_gib'] != EFFECTIVE_CAP_GIB
                or not math.isclose(row['actual_memory_fraction'] * row['device_total_bytes'] / 2**30,
                                    EFFECTIVE_CAP_GIB, abs_tol=1e-6)
                or (rank == 0 and row['result_sha256'] != hashlib.sha256(raw).hexdigest())):
            raise ValueError('All-rank allocator evidence failed')
        contexts.append(hashlib.sha256(context_raw).hexdigest())
    return signature + (tuple(result['manifest']['input_sha256_by_rank']),), dict(
        case=name, result_sha256=hashlib.sha256(raw).hexdigest(), allocator_sha256=contexts,
        median_step_seconds=result['median_step_seconds'], **step_summary(result))


def decisions(rows):
    indexed = {(row['repeat'], row['case']): row for row in rows}
    output = {}
    for candidate, control in (('native_k24', 'native_k36'), ('decoder_k24', 'decoder_k32')):
        pairs = []
        for repeat in (0, 1):
            base, trial = indexed[(repeat, control)], indexed[(repeat, candidate)]
            pairs.append(dict(time_reduction_fraction=1 - trial['median_step_seconds'] / base['median_step_seconds'],
                retry_reduction=base['counter_medians']['num_alloc_retries'] - trial['counter_medians']['num_alloc_retries'],
                sync_reduction=base['counter_medians']['num_sync_all_streams'] - trial['counter_medians']['num_sync_all_streams']))
        faster = all(pair['time_reduction_fraction'] > .02 for pair in pairs)
        output[candidate] = dict(pairs=pairs, worth_further_validation=faster,
            supports_pressure_hypothesis=faster and all(
                pair['retry_reduction'] > 0 and pair['sync_reduction'] > 0 for pair in pairs))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def status(phase, **kw):
        (args.output / 'status.json').write_text(json.dumps(dict(phase=phase, **kw), indent=2))
        print(phase, kw, flush=True)
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    common, extra = source_snapshot(Path.cwd()), helpers()
    frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
    def unchanged():
        if source_snapshot(Path.cwd()) != common or helpers() != extra:
            raise ValueError('Frozen sources changed')
    try:
        status('auditing_profile_evidence')
        if not audit_profiles(Path('artifacts/fsdp_exchange_profiles_v1'))['complete']:
            raise ValueError('Complete profile campaign required')
        verify_frozen(frozen)
        rows, signatures = [], set()
        for repeat, order in enumerate((tuple(CASES), tuple(reversed(CASES)))):
            for name in order:
                label = f'r{repeat}_{name}'
                status('waiting_for_idle_gpus', case=label)
                resources(args.output, label)
                unchanged()
                status('running_screen', case=label)
                log = args.output / f'{label}.log'
                if run_owned(command(name, args.output / label), log, environment(), timeout=1800):
                    raise RuntimeError(f'{label} failed; no fallback')
                unchanged()
                signature, row = audit_run(args.output / label, name)
                signatures.add(signature)
                rows.append(dict(row, repeat=repeat, source=str(args.output / label),
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
                (args.output / 'progress.json').write_text(json.dumps(rows, indent=2))
        if len(signatures) != 1:
            raise ValueError('Unmatched screen workload/source/input signatures')
        unchanged()
        verify_frozen(frozen)
        with (args.output / 'result.json').open('x') as stream:
            json.dump(dict(complete=True, records=rows, decisions=decisions(rows),
                implementation_sources=common, helper_sources=extra, effective_cap_gib=EFFECTIVE_CAP_GIB,
                scope='Unprofiled 12-step ABCD/DCBA screen; not convergence or causal synchronization timing',
                defaults_changed=False, checkpoints_written=False), stream, indent=2)
        status('finished')
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
