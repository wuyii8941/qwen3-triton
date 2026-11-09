"""Independently re-read four-update diagnostics; never infer convergence."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_fsdp_exchange_40 import audit_resources
from scripts.audit_fsdp_exchange_confirmation import loss_comparison
from scripts.diagnose_fsdp_scratch_bounded import audit_short


ORDER = ('native_a', 'native_b', 'bounded_decoder')


def compare_runs(runs):
    native = loss_comparison(runs['native_a'], runs['native_b'])
    candidates = {name: loss_comparison(runs['bounded_decoder'], runs[name]) for name in ORDER[:2]}
    exact = [v['loss'] for v in runs['native_a']['records']] == [v['loss'] for v in runs['native_b']['records']]
    eligible = exact and all(v['within_declared_loss_tolerance'] for v in candidates.values())
    return native, candidates, exact, eligible


def audit(root, *, allow_partial=False):
    path = root / 'result.json'
    complete = path.exists()
    summary = json.loads(path.read_text()) if complete else None
    if complete:
        if (summary['complete'] is not True or summary['ordinary_performance_claim'] is not False
                or summary['convergence_proven'] is not False or summary['default_changed'] is not False
                or json.loads((root / 'status.json').read_text())['phase'] != 'finished'):
            raise ValueError('Finished diagnostic-only study required')
    elif not allow_partial:
        raise ValueError('Explicit partial mode required for incomplete diagnostic')
    rows = summary['records'] if complete else json.loads((root / 'progress.json').read_text())
    if [row['case'] for row in rows] != (list(ORDER) if complete else list(ORDER[:len(rows)])):
        raise ValueError('Wrong ordered diagnostic/prefix')
    runs, signatures, records = {}, set(), []
    for row in rows:
        name = row['case']
        resource_sha = audit_resources(root / (name + '_resources.json'))
        log = (root / (name + '.log')).read_bytes()
        result_path = root / name / 'result.json'
        raw = result_path.read_bytes()
        if (Path(row['source']).resolve() != result_path.resolve()
                or hashlib.sha256(log).hexdigest() != row['log_sha256']
                or hashlib.sha256(raw).hexdigest() != row['sha256']):
            raise ValueError('Raw diagnostic result/log changed')
        result, signature, manifests = audit_short(root / name, name == 'bounded_decoder')
        if manifests != row['rank_manifests']:
            raise ValueError('Actual per-rank context/adapter changed')
        signatures.add(signature)
        snapshot = result['manifest']['implementation_sources']
        encoded = json.dumps(snapshot['files'], sort_keys=True, separators=(',', ':')).encode()
        if (hashlib.sha256(encoded).hexdigest() != snapshot['sha256']
                or result['manifest']['source_sha256'] != snapshot['files']['benchmarks/compare_fsdp_training.py']):
            raise ValueError('Inconsistent source manifest')
        if complete and snapshot != summary['implementation_sources']:
            raise ValueError('Worker and supervisor source differ')
        runs[name] = result
        records.append(dict(case=name, loss=[v['loss'] for v in result['records']],
                            initial_hf_logits_max_abs=result['initial_hf_parity']['max_abs'],
                            deterministic_all_ranks=True, actual_bounded_adapter=name == 'bounded_decoder',
                            resource_sha256=resource_sha))
    if len(signatures) != 1:
        raise ValueError('Matched passing records required')
    native, candidate, exact, eligible = None, {}, None, False
    if complete:
        native, candidate, exact, eligible = compare_runs(runs)
        if (native != summary['native_repeat'] or candidate != summary['candidate_comparisons']
                or exact != summary['native_repeat_exact'] or eligible != summary['eligible_for_quality_preflight']):
            raise ValueError('Diagnostic conclusion differs from raw loss histories')
        for name, digest in summary['helper_sources'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError('Diagnostic helper source changed')
        frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        if frozen['content_sha256'] != summary['frozen_content_sha256']:
            raise ValueError('Frozen input identity differs')
    return dict(complete=complete, partial_diagnostic_only=not complete, raw_records_audited=len(rows),
                records=records, native_repeat=native, native_repeat_exact=exact,
                candidate_comparisons=candidate, eligible_for_quality_preflight=eligible,
                convergence_proven=False, full_gradient_moment_parity_proven=False,
                ordinary_performance_claim=False, default_changed=False,
                scope='Four-update real8B deterministic loss replay only; partial mode cannot qualify quality preflight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = audit(args.root, allow_partial=args.allow_partial)
    with args.output.open('x') as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
