"""All-rank context for the legacy separate-buffer exchange diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

import torch


def hashes():
    names = ('benchmarks/deterministic_fsdp_exchange.py',
             'benchmarks/deterministic_fsdp_comparison.py',
             'benchmarks/compare_fsdp_training.py',
             'benchmarks/fsdp_bf16_exchange_adapter.py',
             'benchmarks/bf16_exchange_reduce.py',
             'scripts/diagnose_fsdp_exchange.py',
             'scripts/audit_fsdp_exchange_confirmation.py',
             'scripts/diagnose_fsdp_1p7b_repeat.py',
             'scripts/fsdp_resource_budget.py')
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    before = hashes()
    runpy.run_path('benchmarks/deterministic_fsdp_comparison.py', run_name='__main__')
    if hashes() != before:
        raise ValueError('Diagnostic sources changed')
    rank = int(os.environ['RANK'])
    context = dict(rank=rank, algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        communication_environment={k: os.environ.get(k) for k in
            ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE')},
        source_sha256=before)
    if not context['algorithms_enabled'] or context['warn_only'] or context['cublas_workspace'] != ':4096:8':
        raise ValueError('Wrong deterministic settings')
    if rank == 0:
        context['result_sha256'] = hashlib.sha256((args.output / 'result.json').read_bytes()).hexdigest()
    with (args.output / f'exchange_deterministic_rank{rank}.json').open('x') as stream:
        json.dump(context, stream, indent=2)


if __name__ == '__main__':
    main()
