"""Audit allocator pairs without conflating allocation policy and training policy."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from scripts.summarize_fsdp_memory import audit_result

CASES = {'native_k35': (35, 2, False), 'expandable_k35': (35, 2, True),
         'native_k30': (30, 0, False), 'expandable_k30': (30, 0, True)}
COUNTERS = ('num_alloc_retries', 'num_ooms', 'num_sync_all_streams',
            'num_device_alloc', 'num_device_free')


def audit_allocator(result, expected):
    allocator = result['allocator']
    if allocator['backend'] != 'native':
        raise ValueError('Native budget-compatible allocator required')
    env = allocator['environment']
    config = f'backend:native,expandable_segments:{expected}'
    if env != {'PYTORCH_ALLOC_CONF': config, 'PYTORCH_CUDA_ALLOC_CONF': None}:
        raise ValueError('Unexpected allocator environment')
    settings = dict(allocator['settings'])
    if settings.pop('expandable_segments') is not expected:
        raise ValueError('Actual allocator setting differs from label')
    if settings.pop('PYTORCH_CUDA_ALLOC_CONF') != config:
        raise ValueError('Actual allocator config differs from environment')
    for row in result['records']:
        if row['peak_reserved_gib'] > result['memory_budget_gib']:
            raise ValueError('Reserved memory exceeds matched budget')
        counters = row['allocator_activity_max_rank']
        for key in COUNTERS:
            value = counters[key]
            if type(value) is not int or value < 0:
                raise ValueError('Supported nonnegative integer counters required')
    return json.dumps(settings, sort_keys=True)


def summarize(root):
    root = Path(root)
    expected_status = [f'{name} returncode=0' for name in CASES]
    if (root/'status.log').read_text().splitlines() != expected_status:
        raise ValueError('Four successful serial cases required; failures need separate review')
    signatures = set(); settings = set(); rows = []
    for name, (keep, every, expandable) in CASES.items():
        path = root/name/'result.json'; raw = path.read_bytes(); result = json.loads(raw)
        signatures.add(audit_result(result, (keep, every, 'full'), vary_allocator=True))
        settings.add(audit_allocator(result, expandable))
        measured = [r for r in result['records'] if not r['warmup']]
        log = root/(name+'.log')
        rows.append(dict(case=name, source=str(path), sha256=hashlib.sha256(raw).hexdigest(),
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            allocator=result['allocator'], median_step_seconds=result['median_step_seconds'],
            peak_allocated_gib=max(r['peak_allocated_gib'] for r in result['records']),
            peak_reserved_gib=max(r['peak_reserved_gib'] for r in result['records']),
            counter_medians={k: statistics.median(r['allocator_activity_max_rank'][k]
                for r in measured) for k in COUNTERS}, measured_records=measured))
    if len(signatures) != 1 or len(settings) != 1:
        raise ValueError('Unmatched workload, sources or other allocator settings')
    comparisons = {f'k{k}': rows[i]['median_step_seconds']/rows[i+1]['median_step_seconds']
                   for i, k in ((0, 35), (2, 30))}
    return dict(complete=True, records=rows, expandable_throughput_ratios=comparisons,
        scope='Single-round matched allocator screen, not repeated or convergence evidence',
        workload_signature=list(next(iter(signatures))))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    report = summarize(args.root)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(report['expandable_throughput_ratios']))


if __name__ == '__main__':
    main()
