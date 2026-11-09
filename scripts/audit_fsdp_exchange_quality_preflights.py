"""Fresh-process audit of completed legacy exchange quality preflights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts.audit_fsdp_exchange_confirmation import audit_launch
from scripts.preflight_fsdp_exchange_quality import MODES, audit_job
from training.fsdp_provenance import source_snapshot


def audit(root, *, allow_partial=False):
    path = root / 'result.json'
    complete = path.exists()
    if not complete and not allow_partial:
        raise ValueError('Complete preflights required; partial mode must be explicit')
    summary = json.loads(path.read_text()) if complete else None
    rows = summary['records'] if complete else json.loads((root / 'progress.json').read_text())
    expected = [prefix + mode for prefix in ('tiny_', 'real_') for mode in MODES]
    if ([r['case'] for r in rows] != expected[:len(rows)] or len(rows) > len(expected)
            or (complete and len(rows) != len(expected))):
        raise ValueError('Wrong ordered preflight coverage')
    original_path = Path('training/fsdp_convergence_protocol_v1.json')
    original = json.loads(original_path.read_text())
    expected_post = copy.deepcopy(original)
    expected_post['validation']['every_updates'] = 2
    if json.loads((root / 'post_update_protocol.json').read_text()) != expected_post:
        raise ValueError('Only post-update validation interval may change')
    records = []
    for row in rows:
        name = row['case']
        prefix, mode = name.split('_', 1)
        tiny = prefix == 'tiny'
        result_path = root / name / 'result.json'
        raw = result_path.read_bytes()
        log = (root / f'{name}.log').read_bytes()
        if (Path(row['source']).resolve() != result_path.resolve()
                or hashlib.sha256(raw).hexdigest() != row['sha256']
                or hashlib.sha256(log).hexdigest() != row['log_sha256']):
            raise ValueError('Preflight raw result/log changed')
        result, manifests = audit_job(root / name, mode,
            original_path if tiny else root / 'post_update_protocol.json', tiny=tiny)
        if (manifests != row['manifest_sha256'] or row['validation'] != result['validation']
                or row['best_parameters_restored'] is not True):
            raise ValueError('Recorded preflight conclusion differs from raw evidence')
        resource_sha = audit_launch(root, name,
            dict(allocator_budget_gib=42., free_margin_gib=2.5, idle_confirmations=3))
        resource = json.loads((root / f'{name}_resources.json').read_text())
        if any(s['mem_available_bytes'] < 160 * 2**30 or s['disk_free_bytes'] < 16 * 2**30 for s in resource['samples']):
            raise ValueError('RAM/disk launch guard violated')
        records.append(dict(case=name, result_sha256=row['sha256'], resource_sha256=resource_sha,
                            manifest_sha256=manifests, validation=result['validation']))
    if complete:
        if (summary['complete'] is not True or summary['preflight_only'] is not True
                or summary['full_quality_completed'] is not False or summary['defaults_changed'] is not False
                or summary['checkpoints_written'] is not False
                or summary['implementation_sources'] != source_snapshot(Path.cwd())
                or summary['original_protocol_sha256'] != hashlib.sha256(original_path.read_bytes()).hexdigest()
                or json.loads((root / 'status.json').read_text())['phase'] != 'finished'):
            raise ValueError('Wrong preflight summary/source/status')
        for name, digest in summary['helper_sources'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError('Preflight helper changed')
        frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        if frozen['content_sha256'] != summary['frozen_content_sha256']:
            raise ValueError('Frozen input identity differs')
    return dict(complete=complete, partial_diagnostic_only=not complete, raw_records_audited=len(rows),
        records=records, all_preflights_passed=complete, full_quality_completed=False,
        defaults_changed=False, checkpoints_written=False,
        scope='Tiny/real two-update full-validation/best-restore integration audit, not convergence or speed evidence')


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
