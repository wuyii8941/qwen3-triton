"""Deterministic diagnostic only; preserve base result and wrapper SHA binding."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from benchmarks.compare_fsdp_scratch_bounded import main as bounded_main, sources


def require_environment(env):
    if env.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Explicit deterministic cuBLAS workspace required')


def hashes():
    name = 'benchmarks/deterministic_fsdp_scratch_bounded.py'
    return dict(sources(), **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    require_environment(os.environ)
    before = hashes()
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    bounded_main()
    if (hashes() != before or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise ValueError('Diagnostic source/settings changed')
    rank = int(os.environ['RANK'])
    path = args.output / f'bounded_manifest_rank{rank}.json'
    record = dict(passed=True, rank=rank, algorithms_enabled=True, warn_only=False,
                  cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                  allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  source_sha256=before, bounded_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  scope='Four-update deterministic diagnostic only; not normal training timing or convergence')
    with (args.output / f'deterministic_context_rank{rank}.json').open('x') as file:
        json.dump(record, file, indent=2)


if __name__ == '__main__':
    main()
