"""Allocator-only envelope; original quality worker and its sidecars unchanged."""
import argparse
import hashlib
import json
import os
import math
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.fsdp_exchange_quality import main as base_main, hashes as base_hashes


ALLOCATOR = 'backend:native,expandable_segments:True'
EFFECTIVE_CAP_GIB = 41.875


def effective_fraction(requested, total_bytes):
    if not math.isclose(requested * total_bytes / 2**30, 42., rel_tol=0, abs_tol=1e-6):
        raise ValueError('Only the explicit nominal 42 GiB experiment is supported')
    return EFFECTIVE_CAP_GIB * 2**30 / total_bytes


def hashes():
    name = 'benchmarks/fsdp_exchange_quality_expandable.py'
    return dict(base_hashes(), **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest()})


def validate_settings(settings, backend, configured, legacy):
    if (backend != 'native' or configured != ALLOCATOR or legacy is not None
            or settings.get('expandable_segments') is not True
            or settings.get('PYTORCH_CUDA_ALLOC_CONF') != ALLOCATOR):
        raise ValueError('Actual native expandable allocator configuration required')


def settings():
    # Small settings-only serialization, never export potentially large snapshots.
    value = torch.cuda.memory._snapshot()['allocator_settings']
    validate_settings(value, torch.cuda.get_allocator_backend(),
                      os.environ.get('PYTORCH_ALLOC_CONF'), os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
    return value


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    before = hashes()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    initial = settings()
    total = torch.cuda.get_device_properties(int(os.environ['LOCAL_RANK'])).total_memory
    original_set = torch.cuda.set_per_process_memory_fraction
    calls = []
    def guarded_set(requested, device=None):
        fraction = effective_fraction(requested, total)
        original_set(fraction, device)
        calls.append(fraction)
    # Process-local, explicit conservative headroom. The unchanged worker's
    # 42 GiB manifest denotes the audited upper bound, not the effective cap.
    with patch('torch.cuda.set_per_process_memory_fraction', guarded_set):
        base_main()
    actual_fraction = torch.cuda.get_per_process_memory_fraction()
    if len(calls) != 1 or not math.isclose(actual_fraction * total / 2**30, EFFECTIVE_CAP_GIB, abs_tol=1e-6):
        raise ValueError('Effective allocator cap was not set as declared')
    final = settings()
    if hashes() != before or final != initial:
        raise ValueError('Allocator/source configuration changed during training')
    rank = int(os.environ['RANK'])
    base_raw = (args.output / f'exchange_quality_rank{rank}.json').read_bytes()
    report = dict(rank=rank, passed=True, backend='native', configured=ALLOCATOR,
        audited_reserved_upper_bound_gib=42., effective_allocator_cap_gib=EFFECTIVE_CAP_GIB,
        actual_memory_fraction=actual_fraction, device_total_bytes=total,
        actual_settings_before=initial, actual_settings_after=final, source_sha256=before,
        base_manifest_sha256=hashlib.sha256(base_raw).hexdigest(),
        scope='Actual runtime allocator settings, not a claim that all workloads avoid OOM')
    if rank == 0:
        report['result_sha256'] = hashlib.sha256((args.output / 'result.json').read_bytes()).hexdigest()
    with (args.output / f'expandable_allocator_rank{rank}.json').open('x') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
