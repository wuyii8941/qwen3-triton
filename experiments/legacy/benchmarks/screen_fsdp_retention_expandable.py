"""Allocator evidence envelope for unprofiled retention/communication screens."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from benchmarks.compare_fsdp_training import main as base_main
from benchmarks.fsdp_exchange_quality_expandable import settings, EFFECTIVE_CAP_GIB


def hashes():
    names = ('benchmarks/screen_fsdp_retention_expandable.py', 'benchmarks/compare_fsdp_training.py',
             'benchmarks/fsdp_exchange_quality_expandable.py', 'benchmarks/fsdp_bf16_exchange_adapter.py',
             'benchmarks/bf16_exchange_reduce.py')
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-budget-gib', type=float, required=True)
    parser.add_argument('--profile', action='store_true')
    args, _ = parser.parse_known_args()
    if args.profile or args.memory_budget_gib != EFFECTIVE_CAP_GIB:
        raise ValueError('Unprofiled, explicit 41.875 GiB screen only')
    before = hashes()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    initial = settings()
    total = torch.cuda.get_device_properties(int(os.environ['LOCAL_RANK'])).total_memory
    # No monkeypatch: the unchanged worker directly sets this exact budget.
    base_main()
    actual = torch.cuda.get_per_process_memory_fraction()
    final = settings()
    if (not math.isclose(actual * total / 2**30, EFFECTIVE_CAP_GIB, abs_tol=1e-6)
            or before != hashes() or final != initial):
        raise ValueError('Source/actual allocator contract changed')
    rank = int(os.environ['RANK'])
    report = dict(rank=rank, passed=True, sources=before, settings_before=initial, settings_after=final,
                  actual_memory_fraction=actual, device_total_bytes=total,
                  effective_cap_gib=EFFECTIVE_CAP_GIB)
    if rank == 0:
        report['result_sha256'] = hashlib.sha256((args.output / 'result.json').read_bytes()).hexdigest()
    with (args.output / f'retention_allocator_rank{rank}.json').open('x') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
