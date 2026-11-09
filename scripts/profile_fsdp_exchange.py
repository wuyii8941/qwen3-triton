"""Serial ABBA rank0 trace diagnostics; profiler is outside steady timing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys

from benchmarks.profile_fsdp_exchange_expandable import hashes as worker_hashes
from benchmarks.fsdp_exchange_quality_expandable import ALLOCATOR, EFFECTIVE_CAP_GIB, validate_settings
from scripts.analyze_fsdp_trace import summarize
from scripts.audit_fsdp_exchange_expandable import audit as audit_quality
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.preflight_fsdp_exchange_expandable import environment, resources
from scripts.summarize_fsdp_memory import audit_result
from training.fsdp_provenance import source_snapshot


ORDER = ('native_a', 'decoder_a', 'decoder_b', 'native_b')


def command(name, output):
    if name not in ORDER:
        raise ValueError('Explicit ABBA case required')
    decoder = name.startswith('decoder')
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
           '--module', 'benchmarks.profile_fsdp_exchange_expandable', '--backend', 'official',
           '--seq', '512', '--batch', '1', '--warmup', '3', '--steps', '12',
           '--keep-full-layers', '32' if decoder else '36', '--checkpoint-every', '1',
           '--memory-budget-gib', '42', '--reference-check', '--profile', '--output', str(output)]
    if decoder:
        cmd += ['--exchange-reduce', 'torch', '--exchange-scope', 'decoder']
    return cmd


def helpers():
    names = ('scripts/profile_fsdp_exchange.py', 'scripts/analyze_fsdp_trace.py',
             'scripts/diagnose_fsdp_1p7b_repeat.py', 'scripts/preflight_fsdp_exchange_expandable.py',
             'scripts/fsdp_resource_budget.py', 'scripts/summarize_fsdp_memory.py')
    return dict(worker_hashes(), **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names})


def audit_case(root, name):
    decoder = name.startswith('decoder')
    raw = (root / 'result.json').read_bytes()
    result = json.loads(raw)
    signature = audit_result(result, (32 if decoder else 36, 1, 'full'), allow_exchange=True)
    if (result['global_parameters'] != 8190735360 or result['world_size'] != 4
            or result['batch'] != 1 or result['seq'] != 512 or result['warmup'] != 3
            or result['steps'] != 12 or 'profile' not in result
            or result['optimization']['exchange_reduce'] != ('torch' if decoder else None)
            or result['optimization']['exchange_scope'] != ('decoder' if decoder else 'all')):
        raise ValueError('Wrong profiled workload')
    expected_env = dict.fromkeys(('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE'))
    if result['communication_environment'] != expected_env:
        raise ValueError('Unmatched NCCL settings')
    counts = result['exchange_callback_counts_by_rank']
    if len(counts) != 4 or any((len(row) != 36 or any(value != 16 for value in row.values()))
                             if decoder else bool(row) for row in counts):
        raise ValueError('Profile-step callback counts must also be included')
    trace_raw = (root / 'trace.json').read_bytes()
    evidence = []
    for rank in range(4):
        context_raw = (root / f'profile_allocator_rank{rank}.json').read_bytes()
        row = json.loads(context_raw)
        for key in ('settings_before', 'settings_after'):
            validate_settings(row[key], 'native', ALLOCATOR, None)
        if (row['rank'] != rank or row['passed'] is not True or row['sources'] != worker_hashes()
                or row['settings_before'] != row['settings_after']
                or row['effective_cap_gib'] != EFFECTIVE_CAP_GIB
                or not math.isclose(row['actual_memory_fraction'] * row['device_total_bytes'] / 2**30,
                                    EFFECTIVE_CAP_GIB, abs_tol=1e-6)
                or (rank == 0 and (row['result.json_sha256'] != hashlib.sha256(raw).hexdigest()
                    or row['trace.json_sha256'] != hashlib.sha256(trace_raw).hexdigest()))):
            raise ValueError('All-rank actual allocator/source evidence failed')
        evidence.append(hashlib.sha256(context_raw).hexdigest())
    report = summarize(json.loads(trace_raw))
    report.update(source_sha256=hashlib.sha256(trace_raw).hexdigest(), source=str(root / 'trace.json'))
    return signature + (tuple(result['manifest']['input_sha256_by_rank']),), dict(
        case=name, source=str(root), result_sha256=hashlib.sha256(raw).hexdigest(),
        allocator_sha256=evidence, unprofiled_median_step_seconds=result['median_step_seconds'],
        trace=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def status(phase, **kw):
        (args.output / 'status.json').write_text(json.dumps(dict(phase=phase, **kw), indent=2))
        print(phase, kw, flush=True)
    def interrupt(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    common, source = source_snapshot(Path.cwd()), helpers()
    frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
    def unchanged():
        if source_snapshot(Path.cwd()) != common or helpers() != source:
            raise ValueError('Frozen implementation changed')
    try:
        status('auditing_completed_quality')
        quality = audit_quality(Path('artifacts/fsdp_exchange_expandable_full_quality_v1'))
        if not quality['full_quality_completed']:
            raise ValueError('Completed full comparison required')
        verify_frozen(frozen)
        rows, signatures = [], set()
        for name in ORDER:
            status('waiting_for_idle_gpus', case=name)
            resources(args.output, name)
            unchanged()
            status('running_profile', case=name)
            if run_owned(command(name, args.output / name), args.output / f'{name}.log', environment(), timeout=1800):
                raise RuntimeError(f'{name} failed; no fallback')
            unchanged()
            signature, row = audit_case(args.output / name, name)
            signatures.add(signature)
            row['log_sha256'] = hashlib.sha256((args.output / f'{name}.log').read_bytes()).hexdigest()
            rows.append(row)
            (args.output / 'progress.json').write_text(json.dumps(rows, indent=2))
        if len(signatures) != 1:
            raise ValueError('Profiles must match source/input/workload signatures')
        verify_frozen(frozen)
        unchanged()
        with (args.output / 'result.json').open('x') as stream:
            json.dump(dict(complete=True, records=rows, implementation_sources=common, helper_sources=source,
                scope='Two rank0 traces per policy, ABBA; diagnostic, not causal overlap ablation or convergence',
                defaults_changed=False, checkpoints_written=False), stream, indent=2)
        status('finished')
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
