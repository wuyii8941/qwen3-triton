"""Independent raw/context/resource audit; partial diagnostics never qualify."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.deterministic_fsdp_exchange import hashes
from scripts.audit_fsdp_exchange_confirmation import audit_launch
from scripts.diagnose_fsdp_exchange import CASES, audit_short, conclusions
from training.fsdp_provenance import source_snapshot


def audit(root, *, allow_partial=False):
    final = root / 'result.json'
    complete = final.exists()
    if not complete and not allow_partial:
        raise ValueError('Complete diagnostic required; partial mode is explicit')
    summary = json.loads(final.read_text()) if complete else None
    rows = summary['records'] if complete else json.loads((root / 'progress.json').read_text())
    if ([r['case'] for r in rows] != list(CASES)[:len(rows)] or len(rows) > len(CASES)
            or (complete and len(rows) != len(CASES))):
        raise ValueError('Wrong diagnostic order or missing cases')
    runs, evidence, signatures = {}, [], []
    for row in rows:
        name = row['case']
        path = root / name / 'result.json'
        raw = path.read_bytes()
        log = (root / f'{name}.log').read_bytes()
        if (Path(row['source']).resolve() != path.resolve()
                or hashlib.sha256(raw).hexdigest() != row['sha256']
                or hashlib.sha256(log).hexdigest() != row['log_sha256']):
            raise ValueError('Raw diagnostic or log changed')
        result, signature, contexts = audit_short(root / name, name)
        if contexts != row['context_sha256']:
            raise ValueError('Context evidence differs')
        resource_sha = audit_launch(root, name,
            dict(allocator_budget_gib=42., free_margin_gib=2.5, idle_confirmations=3))
        resources = json.loads((root / f'{name}_resources.json').read_text())
        if any(s['mem_available_bytes'] < 160 * 2**30 or s['disk_free_bytes'] < 16 * 2**30
               for s in resources['samples']):
            raise ValueError('RAM/disk launch guard violated')
        runs[name] = result
        signatures.append(signature)
        evidence.append(dict(case=name, sha256=row['sha256'], resource_sha256=resource_sha,
                             context_sha256=contexts))
    if signatures and any(s != signatures[0] for s in signatures):
        raise ValueError('Workload or ordered inputs differ')
    checks = conclusions(runs) if complete else None
    if complete:
        if (summary['complete'] is not True or summary['checks'] != checks
                or summary['helper_sources'] != hashes()
                or summary['implementation_sources'] != source_snapshot(Path.cwd())
                or summary['defaults_changed'] is not False
                or summary['losses'] != {n: [v['loss'] for v in r['records']] for n, r in runs.items()}
                or json.loads((root / 'status.json').read_text())['phase'] != 'finished'):
            raise ValueError('Diagnostic summary/status/source mismatch')
        frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        if summary['frozen_content_sha256'] != frozen['content_sha256']:
            raise ValueError('Frozen input identity differs')
    return dict(complete=complete, partial_diagnostic_only=not complete,
        raw_records_audited=len(rows), evidence=evidence, checks=checks,
        eligible_for_quality_preflight={n: c['eligible_for_quality_preflight'] for n, c in checks.items()} if checks else {},
        defaults_changed=False, full_state_equality_proven=False, convergence_proven=False,
        scope='Independent four-update loss/context/source/resource audit, not timing, complete gradients, optimizer states or long quality')


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
