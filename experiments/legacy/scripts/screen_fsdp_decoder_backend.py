"""Fixed-K32 Torch versus Triton local decoder reduction, serial ABBA."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys

from benchmarks.screen_fsdp_retention_expandable import hashes as worker_hashes
from benchmarks.fsdp_exchange_quality_expandable import ALLOCATOR, EFFECTIVE_CAP_GIB, validate_settings
from scripts.audit_fsdp_retention_pressure import audit as audit_pressure
from scripts.audit_fsdp_profiles import step_summary
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.preflight_fsdp_exchange_expandable import environment, resources
from scripts.summarize_fsdp_memory import audit_result
from training.fsdp_provenance import source_snapshot


ORDER = ('torch_a', 'triton_a', 'triton_b', 'torch_b')


def command(name, output):
    if name not in ORDER:
        raise ValueError('Explicit ABBA case required')
    backend = name.split('_')[0]
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
        '--module', 'benchmarks.screen_fsdp_retention_expandable', '--backend', 'official',
        '--seq', '512', '--batch', '1', '--warmup', '3', '--steps', '12',
        '--keep-full-layers', '32', '--checkpoint-every', '1', '--memory-budget-gib', '41.875',
        '--reference-check', '--exchange-reduce', backend, '--exchange-scope', 'decoder', '--output', str(output)]


def gate_command():
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
        'benchmarks/verify_block_fsdp.py', '--backend', 'official', '--bf16', '--exchange-reduce', 'triton',
        '--head-dim', '128', '--seq', '129', '--deterministic', '--keep-full-layers', '2', '--checkpoint-every', '1']


def audit_gate(text):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    if len(rows) != 2 or {row['tied'] for row in rows} != {False, True}:
        raise ValueError('Both tied/untied gates required')
    for row in rows:
        if (row['passed'] is not True or row['backend'] != 'official' or row['world'] != 4
                or row['exchange_reduce'] != 'triton' or row['head_dim'] != 128 or row['seq'] != 129
                or row['steps'] != 3 or row['keep_full_layers'] != 2 or row['checkpoint_every'] != 1
                or row['deterministic'] is not True or row['exchange_callback_calls'] <= 0):
            raise ValueError('Wrong Triton numerical gate contract')
    if sum(row['exchange_callback_calls'] for row in rows) != 27:
        raise ValueError('Incomplete numerical-gate callback execution')
    return rows


def helpers():
    names = ('scripts/screen_fsdp_decoder_backend.py', 'benchmarks/verify_block_fsdp.py',
             'scripts/audit_fsdp_retention_pressure.py', 'scripts/screen_fsdp_retention_pressure.py',
             'scripts/audit_fsdp_profiles.py', 'scripts/preflight_fsdp_exchange_expandable.py',
             'scripts/diagnose_fsdp_1p7b_repeat.py', 'scripts/fsdp_resource_budget.py',
             'scripts/summarize_fsdp_memory.py')
    return dict(worker_hashes(), **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names})


def audit_run(root, name):
    backend = name.split('_')[0]
    raw = (root / 'result.json').read_bytes()
    result = json.loads(raw)
    signature = audit_result(result, (32, 1, 'full'), allow_exchange=True)
    if (result['global_parameters'] != 8190735360 or result['world_size'] != 4
            or result['batch'] != 1 or result['seq'] != 512 or result['warmup'] != 3 or result['steps'] != 12
            or result['memory_budget_gib'] != EFFECTIVE_CAP_GIB or 'profile' in result
            or result['optimization']['exchange_reduce'] != backend
            or result['optimization']['exchange_scope'] != 'decoder'
            or result['communication_environment'] != dict.fromkeys(
                ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE'))):
        raise ValueError('Unmatched backend screen contract')
    counts = result['exchange_callback_counts_by_rank']
    if len(counts) != 4 or any(set(row) != {f'model.layers.{i}' for i in range(36)}
                             or any(value != 15 for value in row.values()) for row in counts):
        raise ValueError('Every decoder callback must execute on every rank')
    contexts = []
    for rank in range(4):
        data = (root / f'retention_allocator_rank{rank}.json').read_bytes()
        row = json.loads(data)
        for key in ('settings_before', 'settings_after'):
            validate_settings(row[key], 'native', ALLOCATOR, None)
        if (row['rank'] != rank or row['passed'] is not True or row['sources'] != worker_hashes()
                or row['settings_before'] != row['settings_after'] or row['effective_cap_gib'] != EFFECTIVE_CAP_GIB
                or not math.isclose(row['actual_memory_fraction'] * row['device_total_bytes'] / 2**30,
                                    EFFECTIVE_CAP_GIB, abs_tol=1e-6)
                or (rank == 0 and row['result_sha256'] != hashlib.sha256(raw).hexdigest())):
            raise ValueError('Actual allocator/source mismatch')
        contexts.append(hashlib.sha256(data).hexdigest())
    return signature + (tuple(result['manifest']['input_sha256_by_rank']),), dict(case=name,
        result_sha256=hashlib.sha256(raw).hexdigest(), allocator_sha256=contexts,
        median_step_seconds=result['median_step_seconds'], **step_summary(result))


def selection(rows):
    by_name = {row['case']: row for row in rows}
    ratios = [1 - by_name[f'triton_{suffix}']['median_step_seconds'] /
              by_name[f'torch_{suffix}']['median_step_seconds'] for suffix in ('a', 'b')]
    return dict(time_reduction_fractions=ratios, worth_quality_validation=all(ratio > .02 for ratio in ratios))


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
            raise ValueError('Frozen source changed')
    try:
        status('auditing_prior_screen')
        if not audit_pressure(Path('artifacts/fsdp_retention_pressure_v1'))['complete']:
            raise ValueError('Prior screen incomplete')
        verify_frozen(frozen)
        status('waiting_for_idle_gpus', case='gate')
        resources(args.output, 'gate')
        unchanged()
        status('running_numerical_gate')
        env = environment()
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        log = args.output / 'gate.log'
        if run_owned(gate_command(), log, env, timeout=600):
            raise RuntimeError('Triton model numerical gate failed; no benchmark')
        unchanged()
        gate = dict(passed=True, records=audit_gate(log.read_text()),
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), source_sha256=extra)
        (args.output / 'gate.json').write_text(json.dumps(gate, indent=2))
        rows, signatures = [], set()
        for name in ORDER:
            status('waiting_for_idle_gpus', case=name)
            resources(args.output, name)
            unchanged()
            status('running_screen', case=name)
            log = args.output / f'{name}.log'
            if run_owned(command(name, args.output / name), log, environment(), timeout=1800):
                raise RuntimeError(f'{name} failed; no fallback')
            unchanged()
            signature, row = audit_run(args.output / name, name)
            signatures.add(signature)
            rows.append(dict(row, source=str(args.output / name),
                             log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
            (args.output / 'progress.json').write_text(json.dumps(rows, indent=2))
        if len(signatures) != 1:
            raise ValueError('Unmatched implementation/input/workload')
        verify_frozen(frozen)
        unchanged()
        with (args.output / 'result.json').open('x') as stream:
            json.dump(dict(complete=True, gate=gate, records=rows, selection=selection(rows),
                implementation_sources=common, helper_sources=extra, defaults_changed=False,
                full_quality_completed=False, scope='Fixed K32 ABBA local reduction backend screen, not convergence'), stream, indent=2)
        status('finished')
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
