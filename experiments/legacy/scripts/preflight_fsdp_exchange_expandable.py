"""Matched allocator preflights, retaining the failed non-expandable campaign."""
import argparse
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

from benchmarks.fsdp_exchange_quality_expandable import ALLOCATOR, EFFECTIVE_CAP_GIB, hashes as worker_hashes, validate_settings
from scripts.audit_fsdp_exchange_quality_preflights import audit as audit_prior_preflights
from scripts.audit_fsdp_exchange_deterministic import audit as audit_diagnostic
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.fsdp_resource_budget import budget_ready
from scripts.preflight_fsdp_exchange_quality import command as base_command, audit_job as audit_base
from training.fsdp_provenance import source_snapshot


def command(output, mode, protocol, *, tiny=False, preflight=True):
    if mode not in ('native', 'decoder') or (tiny and not preflight):
        raise ValueError('Only native/decoder with explicit tiny cutoff')
    cmd = base_command(output, mode, protocol, tiny=tiny)
    cmd[cmd.index('benchmarks.fsdp_exchange_quality')] = 'benchmarks.fsdp_exchange_quality_expandable'
    if not preflight:
        i = cmd.index('--preflight-steps')
        del cmd[i:i + 2]
    return cmd


def audit_allocator(directory):
    contexts = []
    for rank in range(4):
        path = directory / f'expandable_allocator_rank{rank}.json'
        raw = path.read_bytes()
        row = json.loads(raw)
        for key in ('actual_settings_before', 'actual_settings_after'):
            validate_settings(row[key], row['backend'], row['configured'], None)
        if (row['rank'] != rank or row['passed'] is not True or row['source_sha256'] != worker_hashes()
                or row['audited_reserved_upper_bound_gib'] != 42.
                or row['effective_allocator_cap_gib'] != EFFECTIVE_CAP_GIB
                or not math.isclose(row['actual_memory_fraction'] * row['device_total_bytes'] / 2**30, EFFECTIVE_CAP_GIB, abs_tol=1e-6)
                or row['actual_settings_before'] != row['actual_settings_after']
                or row['base_manifest_sha256'] != hashlib.sha256((directory / f'exchange_quality_rank{rank}.json').read_bytes()).hexdigest()
                or (rank == 0 and row['result_sha256'] != hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest())):
            raise ValueError('All-rank actual allocator evidence failed')
        contexts.append(hashlib.sha256(raw).hexdigest())
    return contexts


def resources(output, name):
    samples = []
    while len(samples) < 3:
        values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        readings = [tuple(int(v.strip()) for v in line.split(',')) for line in values.splitlines()]
        ram = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        disk = shutil.disk_usage(Path.cwd()).free
        if budget_ready(readings, allocator_gib=42, free_margin_gib=2.5) and ram >= 160 * 2**30 and disk >= 16 * 2**30:
            samples.append(dict(unix_seconds=time.time(), readings=readings[:4], mem_available_bytes=ram, disk_free_bytes=disk))
        else:
            samples = []
        if len(samples) < 3:
            time.sleep(15)
    (output / f'{name}_resources.json').write_text(json.dumps(dict(allocator_budget_gib=42.,
        free_margin_gib=2.5, idle_confirmations=3, samples=samples), indent=2))


def environment():
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTORCH_ALLOC_CONF=ALLOCATOR)
    for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE',
                'PYTORCH_CUDA_ALLOC_CONF', 'CUBLAS_WORKSPACE_CONFIG'):
        env.pop(key, None)
    return env


def hashes():
    names = ('scripts/preflight_fsdp_exchange_expandable.py', 'scripts/preflight_fsdp_exchange_quality.py',
        'scripts/audit_fsdp_exchange_quality_preflights.py', 'scripts/audit_fsdp_exchange_deterministic.py',
        'scripts/diagnose_fsdp_1p7b_repeat.py', 'scripts/fsdp_resource_budget.py')
    return dict(worker_hashes(), **{n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names})


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
    common, before = source_snapshot(Path.cwd()), hashes()
    protocol = Path('training/fsdp_convergence_protocol_v1.json')
    protocol_raw = protocol.read_bytes()
    post = json.loads(protocol_raw)
    post['validation']['every_updates'] = 2
    post_path = args.output / 'post_update_protocol.json'
    post_path.write_text(json.dumps(post, indent=2))
    post_raw = post_path.read_bytes()
    def unchanged():
        if (hashes() != before or source_snapshot(Path.cwd()) != common or protocol.read_bytes() != protocol_raw
                or post_path.read_bytes() != post_raw):
            raise ValueError('Source/protocol changed')
    try:
        status('auditing_prior_gates')
        if (not audit_prior_preflights(Path('artifacts/fsdp_exchange_quality_preflights_v1'))['all_preflights_passed']
                or not audit_diagnostic(Path('artifacts/fsdp_exchange_deterministic_v1'))['eligible_for_quality_preflight']['decoder']):
            raise ValueError('Prior integration/numerical gates required')
        frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        verify_frozen(frozen)
        rows = []
        for tiny in (True, False):
            for mode in ('native', 'decoder'):
                name = ('tiny_' if tiny else 'real_') + mode
                status('waiting_for_idle_gpus', case=name)
                resources(args.output, name)
                unchanged()
                status('running_preflight', case=name)
                selected = protocol if tiny else post_path
                log = args.output / f'{name}.log'
                if run_owned(command(args.output / name, mode, selected, tiny=tiny), log, environment(), timeout=1800):
                    raise RuntimeError(f'{name} failed; no fallback')
                unchanged()
                result, manifests = audit_base(args.output / name, mode, selected, tiny=tiny)
                allocators = audit_allocator(args.output / name)
                rows.append(dict(case=name, source=str(args.output / name / 'result.json'),
                    sha256=hashlib.sha256((args.output / name / 'result.json').read_bytes()).hexdigest(),
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), manifest_sha256=manifests,
                    allocator_sha256=allocators, validation=result['validation']))
                (args.output / 'progress.json').write_text(json.dumps(rows, indent=2))
        verify_frozen(frozen)
        unchanged()
        (args.output / 'result.json').write_text(json.dumps(dict(complete=True, preflight_only=True,
            full_quality_completed=False, records=rows, implementation_sources=common, helper_sources=before,
            allocator=ALLOCATOR, protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),
            effective_allocator_cap_gib=EFFECTIVE_CAP_GIB,
            frozen_content_sha256=frozen['content_sha256'], defaults_changed=False, checkpoints_written=False), indent=2))
        status('finished')
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
