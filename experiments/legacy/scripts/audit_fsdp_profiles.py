"""Re-read all four raw profile jobs; do not infer causal or quality gains."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from scripts.profile_fsdp_exchange import ORDER, audit_case, helpers
from scripts.fsdp_resource_budget import budget_ready
from training.fsdp_provenance import source_snapshot


def step_summary(result):
    measured = [row for row in result['records'] if not row['warmup']]
    counters = measured[0]['allocator_activity_max_rank']
    return dict(peak_allocated_gib=max(row['peak_allocated_gib'] for row in measured),
                peak_reserved_gib=max(row['peak_reserved_gib'] for row in measured),
                counter_medians={key: statistics.median(row['allocator_activity_max_rank'][key] for row in measured)
                    if all(row['allocator_activity_max_rank'][key] is not None for row in measured) else None
                    for key in counters})


def audit(root):
    result_raw = (root / 'result.json').read_bytes()
    result = json.loads(result_raw)
    if (result.get('complete') is not True or [row['case'] for row in result['records']] != list(ORDER)
            or result.get('defaults_changed') is not False or result.get('checkpoints_written') is not False
            or json.loads((root / 'status.json').read_text())['phase'] != 'finished'):
        raise ValueError('Complete ABBA profile campaign required')
    if result['implementation_sources'] != source_snapshot(Path.cwd()) or result['helper_sources'] != helpers():
        raise ValueError('Source identity changed')
    rows, signatures = [], set()
    for recorded in result['records']:
        name = recorded['case']
        signature, actual = audit_case(root / name, name)
        signatures.add(signature)
        actual['log_sha256'] = hashlib.sha256((root / f'{name}.log').read_bytes()).hexdigest()
        if actual != recorded:
            raise ValueError('Stored result differs from raw trace/job evidence')
        resource_raw = (root / f'{name}_resources.json').read_bytes()
        resource = json.loads(resource_raw)
        if (resource['allocator_budget_gib'] != 42 or resource['free_margin_gib'] != 2.5
                or resource['idle_confirmations'] != 3 or len(resource['samples']) != 3
                or any(not budget_ready(sample['readings'], allocator_gib=42, free_margin_gib=2.5)
                    or sample['mem_available_bytes'] < 160 * 2**30
                    or sample['disk_free_bytes'] < 16 * 2**30 for sample in resource['samples'])):
            raise ValueError('Invalid launch resource samples')
        worker = json.loads((root / name / 'result.json').read_text())
        rows.append(dict(case=name, median_step_seconds=actual['unprofiled_median_step_seconds'],
            **step_summary(worker), trace_sha256=actual['trace']['source_sha256'],
            trace_intervals={device: {key: value for key, value in record.items() if key != 'kernels'}
                             for device, record in actual['trace']['devices'].items()},
            resource_sha256=hashlib.sha256(resource_raw).hexdigest()))
    if len(signatures) != 1:
        raise ValueError('Unmatched workload/source/input signatures')
    by_name = {row['case']: row for row in rows}
    return dict(complete=True, raw_jobs_audited=4, records=rows,
        decoder_time_reduction_fraction_by_order=[1 - by_name[f'decoder_{suffix}']['median_step_seconds'] /
            by_name[f'native_{suffix}']['median_step_seconds'] for suffix in ('a', 'b')],
        campaign_result_sha256=hashlib.sha256(result_raw).hexdigest(),
        statistical_significance_proven=False, causal_overlap_speedup_proven=False,
        scope='Rank0 trace interval diagnostics and separate unprofiled step timing; not convergence or wire bandwidth')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
