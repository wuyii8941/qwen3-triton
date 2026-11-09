"""Explicit bounded-adapter entry around the unchanged training benchmark.

Base result.json is preserved verbatim. It is NOT sufficient evidence alone:
bounded_manifest_rank*.json overrides its legacy debug-disabled text and pins
the actually injected adapter. No installed Torch or base source is modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from benchmarks.bf16_scratch_bounded import BoundedScratchExchangeReduceScatter
from benchmarks.compare_fsdp_training import main as base_main


def sources():
    names = ('benchmarks/compare_fsdp_scratch_bounded.py', 'benchmarks/compare_fsdp_training.py',
             'benchmarks/bf16_scratch_bounded.py', 'benchmarks/bf16_scratch_validation.py',
             'benchmarks/bf16_scratch_pack.py', 'benchmarks/fsdp_scratch_exchange_adapter.py',
             'benchmarks/bf16_exchange_reduce.py', 'benchmarks/fsdp_bf16_exchange_adapter.py')
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}


def configure(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--bounded-exchange-scope', choices=['head', 'decoder', 'all'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['official'], required=True)
    args, rest = parser.parse_known_args(argv)
    if any(token.startswith('--exchange-reduce') or token.startswith('--exchange-scope') for token in rest):
        raise ValueError('Use explicit bounded-exchange-scope, not the original adapter flags')
    base = rest + ['--output', str(args.output), '--backend', 'official']
    if args.bounded_exchange_scope:
        base += ['--exchange-reduce', 'triton', '--exchange-scope', args.bounded_exchange_scope]
    return args, base


def main():
    args, base_argv = configure(sys.argv[1:])
    before = sources()
    instances = []
    def factory(*, original_bf16, accumulation=1, backend='triton', debug=False):
        # The legacy benchmark requests debug=False. We deliberately override
        # that request and include all safety checks in the measured step.
        instance = BoundedScratchExchangeReduceScatter(original_bf16=original_bf16,
            accumulation=accumulation, backend=backend, debug=True)
        instances.append(instance)
        return instance
    saved = sys.argv
    try:
        sys.argv = [str(Path('benchmarks/compare_fsdp_training.py')), *base_argv]
        with patch('benchmarks.fsdp_bf16_exchange_adapter.BF16ExchangeReduceScatter', factory):
            base_main()
    finally:
        sys.argv = saved
    if sources() != before:
        raise ValueError('Wrapper or actual adapter source changed during measurement')
    calls = [instance.calls for instance in instances]
    if args.bounded_exchange_scope:
        if not calls or any(count <= 0 for count in calls):
            raise ValueError('Actual bounded callback did not execute')
    elif calls:
        raise ValueError('Native wrapper control unexpectedly used custom communication')
    rank = int(os.environ['RANK'])
    record = dict(rank=rank, passed=True, bounded_scope=args.bounded_exchange_scope,
                  actual_adapter='BoundedScratchExchangeReduceScatter' if calls else 'native',
                  actual_value_validation_enabled=bool(calls), actual_callback_counts=calls,
                  input_mutation=bool(calls), source_sha256=before,
                  legacy_base_debug_contract_applicable=not bool(calls),
                  scope='Mandatory wrapper evidence; base result alone omits actual adapter identity/validation policy')
    if rank == 0:
        record['base_result_sha256'] = hashlib.sha256((args.output / 'result.json').read_bytes()).hexdigest()
    with (args.output / f'bounded_manifest_rank{rank}.json').open('x') as file:
        json.dump(record, file, indent=2)


if __name__ == '__main__':
    main()
