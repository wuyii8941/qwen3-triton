"""Unchanged step benchmark with the full-quality experiment's allocator cap."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.compare_fsdp_training import main as base_main
from benchmarks.fsdp_exchange_quality_expandable import effective_fraction, settings, EFFECTIVE_CAP_GIB


def hashes():
    names = ('benchmarks/profile_fsdp_exchange_expandable.py', 'benchmarks/compare_fsdp_training.py',
             'benchmarks/fsdp_exchange_quality_expandable.py', 'benchmarks/fsdp_bf16_exchange_adapter.py',
             'benchmarks/bf16_exchange_reduce.py')
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    args, _ = parser.parse_known_args()
    if not args.profile:
        raise ValueError('This entry requires an explicitly separate profile step')
    before = hashes()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    initial = settings()
    total = torch.cuda.get_device_properties(int(os.environ['LOCAL_RANK'])).total_memory
    original = torch.cuda.set_per_process_memory_fraction
    calls = []
    def capped(requested, device=None):
        fraction = effective_fraction(requested, total)
        original(fraction, device)
        calls.append(fraction)
    with patch('torch.cuda.set_per_process_memory_fraction', capped):
        base_main()
    actual = torch.cuda.get_per_process_memory_fraction()
    final = settings()
    if (len(calls) != 1 or not math.isclose(actual * total / 2**30, EFFECTIVE_CAP_GIB, abs_tol=1e-6)
            or before != hashes() or final != initial):
        raise ValueError('Source/actual allocator contract changed')
    rank = int(os.environ['RANK'])
    report = dict(rank=rank, passed=True, sources=before, settings_before=initial, settings_after=final,
                  actual_memory_fraction=actual, device_total_bytes=total,
                  effective_cap_gib=EFFECTIVE_CAP_GIB)
    if rank == 0:
        for name in ('result.json', 'trace.json'):
            report[name + '_sha256'] = hashlib.sha256((args.output / name).read_bytes()).hexdigest()
    with (args.output / f'profile_allocator_rank{rank}.json').open('x') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
