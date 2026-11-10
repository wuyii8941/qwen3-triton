"""Legacy separate-buffer exchange in the unchanged quality training loop.

This matches the performance worker's debug=False adapter, not the checked
bounded-scratch adapter. All rank sidecars are mandatory evidence.
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

from benchmarks.fsdp_bf16_exchange_adapter import BF16ExchangeReduceScatter
from training.fsdp_convergence_job import main as quality_main
from training.official_fsdp import shard_official_qwen as original_shard


def configure(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', choices=['official'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-budget-gib', type=float, required=True)
    parser.add_argument('--exchange-backend', choices=['torch', 'triton'])
    parser.add_argument('--exchange-scope', choices=['decoder', 'head'])
    parser.add_argument('--protocol', type=Path, default=Path('training/fsdp_convergence_protocol_v1.json'))
    args, rest = parser.parse_known_args(argv)
    if args.memory_budget_gib != 42:
        raise ValueError('This experiment requires its measured 42 GiB budget')
    if (args.exchange_backend, args.exchange_scope) not in ((None, None), ('torch', 'decoder'), ('triton', 'head')):
        raise ValueError('Native, Torch decoder or Triton head only')
    if any(v.startswith(('--exchange-', '--bounded-')) for v in rest):
        raise ValueError('Unsupported additional communication option')
    protocol = os.environ.get('NCCL_PROTO')
    if (protocol not in (None, 'Simple') or (args.exchange_scope == 'head' and protocol != 'Simple')
            or (args.exchange_scope == 'decoder' and protocol is not None)
            or any(os.environ.get(k) for k in ('NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE'))):
        raise ValueError('Communication environment differs from measured candidate')
    return args, rest + ['--backend', 'official', '--output', str(args.output), '--protocol', str(args.protocol)]


def hashes():
    names = ('benchmarks/fsdp_exchange_quality.py', 'benchmarks/fsdp_bf16_exchange_adapter.py',
        'benchmarks/bf16_exchange_reduce.py', 'training/fsdp_convergence_job.py',
        'training/official_fsdp.py', 'training/fsdp_best_parameters.py', 'training/fsdp_validation.py')
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}


def install(model, backend, scope, *, fsdp_type):
    if backend is None:
        return []
    if scope == 'head' and model.model.embed_tokens.weight is model.lm_head.weight:
        raise ValueError('Head-only exchange does not support tied embedding FSDP groups')
    units = ([(f'model.layers.{i}', layer) for i, layer in enumerate(model.model.layers)]
             if scope == 'decoder' else [('lm_head', model.lm_head)])
    callbacks = []
    for name, unit in units:
        if not isinstance(unit, fsdp_type):
            raise ValueError(f'{name} is not an actual FSDP unit')
        callback = BF16ExchangeReduceScatter(original_bf16=True, accumulation=1, backend=backend, debug=False)
        unit.set_custom_reduce_scatter(callback)
        callbacks.append((name, callback))
    if not callbacks:
        raise ValueError('Empty exchange unit selection')
    return callbacks


def main():
    args, base = configure(sys.argv[1:])
    before = hashes()
    protocol_sha = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    total = torch.cuda.get_device_properties().total_memory
    fraction = args.memory_budget_gib * 2**30 / total
    if torch.cuda.get_allocator_backend() != 'native' or not 0 < fraction < 1:
        raise ValueError('Native CUDA allocator with explicit sub-capacity budget required')
    torch.cuda.set_per_process_memory_fraction(fraction)
    callbacks = []
    def shard(*positional, **keyword):
        from torch.distributed.fsdp import FSDPModule
        model = original_shard(*positional, **keyword)
        callbacks.extend(install(model, args.exchange_backend, args.exchange_scope, fsdp_type=FSDPModule))
        return model
    saved = sys.argv
    try:
        sys.argv = ['training/fsdp_convergence_job.py', *base]
        with patch('training.official_fsdp.shard_official_qwen', shard), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            quality_main()
    finally:
        sys.argv = saved
    if hashes() != before or hashlib.sha256(args.protocol.read_bytes()).hexdigest() != protocol_sha:
        raise ValueError('Quality source or protocol changed')
    counts = {name: callback.calls for name, callback in callbacks}
    if args.exchange_backend and (not counts or any(v <= 0 for v in counts.values())):
        raise ValueError('Selected communication callbacks did not execute')
    rank = int(os.environ['RANK'])
    manifest = dict(passed=True, rank=rank, actual_adapter='BF16ExchangeReduceScatter' if counts else 'native',
        exchange_backend=args.exchange_backend, exchange_scope=args.exchange_scope,
        callback_counts=counts, actual_value_validation_enabled=False,
        memory_budget_gib=42., attention='forced Flash SDPA', source_sha256=before,
        protocol_file_sha256=protocol_sha,
        communication_environment={k: os.environ.get(k) for k in
            ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE')},
        scope='Separate-buffer adapter matching measured debug=False path; not bounded scratch or a convergence claim')
    if rank == 0:
        raw = (args.output / 'result.json').read_bytes()
        result = json.loads(raw)
        if not result['complete'] or any(v != len(result['training']) for v in counts.values()):
            raise ValueError('Incomplete training or wrong callback count')
        manifest['base_result_sha256'] = hashlib.sha256(raw).hexdigest()
    with (args.output / f'exchange_quality_rank{rank}.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == '__main__':
    main()
