"""Independent completed-stage audit, including actual allocator snapshots."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from benchmarks.fsdp_exchange_quality_expandable import ALLOCATOR, EFFECTIVE_CAP_GIB
from scripts.audit_fsdp_exchange_confirmation import audit_launch
from scripts.preflight_fsdp_exchange_expandable import audit_allocator
from scripts.preflight_fsdp_exchange_quality import audit_job
from training.fsdp_provenance import source_snapshot


def audit(root, *, allow_partial=False):
    final = root / 'result.json'
    complete = final.exists()
    if not complete and not allow_partial:
        raise ValueError('Complete allocator preflights required')
    summary = json.loads(final.read_text()) if complete else None
    rows = summary['records'] if complete else json.loads((root / 'progress.json').read_text())
    expected = ['tiny_native', 'tiny_decoder', 'real_native', 'real_decoder']
    if ([r['case'] for r in rows] != expected[:len(rows)] or len(rows) > 4 or (complete and len(rows) != 4)):
        raise ValueError('Wrong ordered allocator preflight coverage')
    protocol = Path('training/fsdp_convergence_protocol_v1.json')
    post_path = root / 'post_update_protocol.json'
    expected_post = copy.deepcopy(json.loads(protocol.read_text()))
    expected_post['validation']['every_updates'] = 2
    if json.loads(post_path.read_text()) != expected_post:
        raise ValueError('Unintended training protocol change')
    evidence = []
    for row in rows:
        name = row['case']
        prefix, mode = name.split('_', 1)
        tiny = prefix == 'tiny'
        path = root / name / 'result.json'
        if (Path(row['source']).resolve() != path.resolve()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']
                or hashlib.sha256((root / f'{name}.log').read_bytes()).hexdigest() != row['log_sha256']):
            raise ValueError('Allocator preflight raw result/log changed')
        result, manifests = audit_job(root / name, mode, protocol if tiny else post_path, tiny=tiny)
        contexts = audit_allocator(root / name)
        if (row['manifest_sha256'] != manifests or row['allocator_sha256'] != contexts
                or row['validation'] != result['validation']):
            raise ValueError('Allocator/integration/history evidence differs')
        resource_sha = audit_launch(root, name,
            dict(allocator_budget_gib=42., free_margin_gib=2.5, idle_confirmations=3))
        resources = json.loads((root / f'{name}_resources.json').read_text())
        if any(s['mem_available_bytes'] < 160 * 2**30 or s['disk_free_bytes'] < 16 * 2**30 for s in resources['samples']):
            raise ValueError('RAM/disk launch guard violated')
        evidence.append(dict(case=name, result_sha256=row['sha256'], resource_sha256=resource_sha,
            manifest_sha256=manifests, allocator_sha256=contexts, validation=result['validation']))
    if complete:
        if (summary['complete'] is not True or summary['preflight_only'] is not True
                or summary['full_quality_completed'] is not False or summary['allocator'] != ALLOCATOR
                or summary['effective_allocator_cap_gib'] != EFFECTIVE_CAP_GIB
                or summary['implementation_sources'] != source_snapshot(Path.cwd())
                or summary['protocol_sha256'] != hashlib.sha256(protocol.read_bytes()).hexdigest()
                or summary['defaults_changed'] is not False or summary['checkpoints_written'] is not False
                or json.loads((root / 'status.json').read_text())['phase'] != 'finished'):
            raise ValueError('Wrong allocator preflight summary/status/source')
        for name, digest in summary['helper_sources'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError('Allocator preflight helper changed')
        frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        if frozen['content_sha256'] != summary['frozen_content_sha256']:
            raise ValueError('Frozen input identity differs')
    return dict(complete=complete, partial_diagnostic_only=not complete, raw_records_audited=len(rows),
        evidence=evidence, allocator=ALLOCATOR, all_preflights_passed=complete, full_quality_completed=False,
        effective_allocator_cap_gib=EFFECTIVE_CAP_GIB, audited_reserved_upper_bound_gib=42.,
        long_run_oom_fixed_proven=False, defaults_changed=False, checkpoints_written=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = audit(args.root, allow_partial=args.allow_partial)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
