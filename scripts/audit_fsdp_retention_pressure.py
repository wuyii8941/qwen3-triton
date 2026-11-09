"""Independent raw-result audit for the allocator-pressure retention screen."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.screen_fsdp_retention_pressure import CASES, audit_run, decisions, helpers
from scripts.fsdp_resource_budget import budget_ready
from training.fsdp_provenance import source_snapshot


def audit_resources(resource):
    if (resource['allocator_budget_gib'] != 42 or resource['free_margin_gib'] != 2.5
            or resource['idle_confirmations'] != 3 or len(resource['samples']) != 3
            or any(not budget_ready(row['readings'], allocator_gib=42, free_margin_gib=2.5)
                or row['mem_available_bytes'] < 160 * 2**30 or row['disk_free_bytes'] < 16 * 2**30
                for row in resource['samples'])):
        raise ValueError('Resource guard evidence failed')


def audit(root, *, allow_partial=False):
    final = root / 'result.json'
    complete = final.exists()
    if complete:
        raw = final.read_bytes()
        result = json.loads(raw)
        if (result.get('complete') is not True or result['effective_cap_gib'] != 41.875
                or result['defaults_changed'] is not False or result['checkpoints_written'] is not False
                or result['implementation_sources'] != source_snapshot(Path.cwd())
                or result['helper_sources'] != helpers()
                or json.loads((root / 'status.json').read_text())['phase'] != 'finished'):
            raise ValueError('Complete matched-source screen required')
        recorded = result['records']
    elif allow_partial:
        recorded = json.loads((root / 'progress.json').read_text())
    else:
        raise ValueError('Screen incomplete')
    expected = [(repeat, name) for repeat, order in enumerate((tuple(CASES), tuple(reversed(CASES))))
                for name in order]
    order = [(row['repeat'], row['case']) for row in recorded]
    if not order or order != expected[:len(order)] or (complete and len(order) != 8):
        raise ValueError('Exact ABCD/DCBA order and coverage required')
    signatures, evidence = set(), []
    for row in recorded:
        name, repeat = row['case'], row['repeat']
        label = f'r{repeat}_{name}'
        signature, actual = audit_run(root / label, name)
        signatures.add(signature)
        actual.update(repeat=repeat, source=str(root / label),
                      log_sha256=hashlib.sha256((root / f'{label}.log').read_bytes()).hexdigest())
        if actual != row:
            raise ValueError('Summary disagrees with raw worker/context/log evidence')
        # Verify current source even for partial audits, not only stored labels.
        worker = json.loads((root / label / 'result.json').read_text())
        if worker['manifest']['implementation_sources'] != source_snapshot(Path.cwd()):
            raise ValueError('Partial source mismatch')
        resources_raw = (root / f'{label}_resources.json').read_bytes()
        audit_resources(json.loads(resources_raw))
        evidence.append(dict(row, resource_sha256=hashlib.sha256(resources_raw).hexdigest()))
    if len(signatures) != 1:
        raise ValueError('Unmatched workload/input/source signatures')
    selection = decisions(recorded) if complete else None
    if complete and selection != result['decisions']:
        raise ValueError('Decision does not follow raw measurements')
    return dict(complete=complete, partial_diagnostic_only=not complete,
                raw_jobs_audited=len(recorded), records=evidence, decisions=selection,
                full_quality_completed=False, defaults_changed=False,
                scope='Short unprofiled same-cap screen; not causal synchronization timing or convergence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report = audit(args.root, allow_partial=args.allow_partial)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
