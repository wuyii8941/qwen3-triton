"""Quality-job wrapper: same frozen protocol, explicit budget and adapter.

Not launched automatically by a performance win. Preflight and full plateau
jobs must retain base result plus every rank's quality adapter manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from benchmarks.bf16_scratch_bounded import BoundedScratchExchangeReduceScatter
from benchmarks.compare_fsdp_scratch_bounded import sources
from training.fsdp_convergence_job import main as quality_main
from training.official_fsdp import shard_official_qwen as original_shard


def configure(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', choices=['official'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-budget-gib', type=float, required=True)
    parser.add_argument('--bounded-exchange-scope', choices=['decoder'])
    parser.add_argument('--protocol', type=Path, default=Path('training/fsdp_convergence_protocol_v1.json'))
    args, rest = parser.parse_known_args(argv)
    if any(token.startswith('--exchange-') for token in rest):
        raise ValueError('Only explicit bounded decoder mode is supported')
    base = rest + ['--backend', 'official', '--output', str(args.output), '--protocol', str(args.protocol)]
    return args, base


def hashes():
    extra = ('benchmarks/fsdp_scratch_bounded_quality.py', 'training/fsdp_convergence_job.py',
             'training/official_fsdp.py', 'training/fsdp_validation.py', 'training/fsdp_best_parameters.py')
    return dict(sources(), **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in extra})


def main():
    args, base = configure(sys.argv[1:])
    before = hashes()
    protocol_hash = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    total = torch.cuda.get_device_properties().total_memory
    fraction = args.memory_budget_gib * 2**30 / total
    if torch.cuda.get_allocator_backend() != 'native' or not 0 < fraction < 1:
        raise ValueError('Explicit positive native allocator budget below device capacity required')
    torch.cuda.set_per_process_memory_fraction(fraction)
    callbacks = []
    def shard(*positional, **keyword):
        model = original_shard(*positional, **keyword)
        if args.bounded_exchange_scope:
            from torch.distributed.fsdp import FSDPModule
            for name, unit in model.named_modules():
                if name.startswith('model.layers.') and isinstance(unit, FSDPModule):
                    callback = BoundedScratchExchangeReduceScatter(original_bf16=True,
                        accumulation=1, backend='triton', debug=True)
                    unit.set_custom_reduce_scatter(callback)
                    callbacks.append((name, callback))
            if len(callbacks) != len(model.model.layers):
                raise ValueError('Every decoder FSDP unit must use the actual bounded adapter')
        return model
    saved = sys.argv
    try:
        sys.argv = ['training/fsdp_convergence_job.py', *base]
        # Only a process-local testable factory dependency is replaced. The
        # frozen training implementation/protocol and installed Torch stay intact.
        with patch('training.official_fsdp.shard_official_qwen', shard), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            quality_main()
    finally:
        sys.argv = saved
    if hashes() != before or hashlib.sha256(args.protocol.read_bytes()).hexdigest() != protocol_hash:
        raise ValueError('Quality adapter or protocol changed during job')
    counts = {name: callback.calls for name, callback in callbacks}
    if args.bounded_exchange_scope and (not counts or any(v <= 0 for v in counts.values())):
        raise ValueError('Actual bounded quality callback did not execute')
    rank = int(os.environ['RANK'])
    manifest = dict(passed=True, rank=rank, bounded_scope=args.bounded_exchange_scope,
                    actual_adapter='BoundedScratchExchangeReduceScatter' if counts else 'native',
                    actual_callback_counts=counts, actual_value_validation_enabled=bool(counts),
                    memory_budget_gib=args.memory_budget_gib, attention='forced Flash SDPA',
                    source_sha256=before, protocol_file_sha256=protocol_hash,
                    scope='Mandatory adapter provenance; convergence must be established from full base protocol/history/test')
    if rank == 0:
        raw = (args.output / 'result.json').read_bytes()
        manifest['base_result_sha256'] = hashlib.sha256(raw).hexdigest()
        result = json.loads(raw)
        if not result['complete'] or any(v != len(result['training']) for v in counts.values()):
            raise ValueError('Incomplete quality job or unexpected callback update count')
    with (args.output / f'quality_adapter_rank{rank}.json').open('x') as file:
        json.dump(manifest, file, indent=2)


if __name__ == '__main__':
    main()
