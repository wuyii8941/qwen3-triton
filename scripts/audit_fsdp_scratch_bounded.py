"""Fresh-process raw/sidecar/resource audit. Partial mode never selects winners."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_fsdp_exchange_40 import audit_resources
from scripts.audit_fsdp_exchange_confirmation import loss_comparison
from scripts.screen_fsdp_scratch_bounded import CASES, audit_run


def recompute_decisions(rows):
    indexed = {(row['repeat'], row['case']): row for row in rows}
    output = {}
    for candidate in ('bounded_head_k34', 'bounded_decoder_k34'):
        ratios = []
        for repeat in (0, 1):
            control_times = [indexed[(repeat, name)]['median_step_seconds']
                             for name in ('native_k32', 'native_k34')
                             if indexed[(repeat, name)]['status'] == 'passed']
            row = indexed[(repeat, candidate)]
            ratio = (min(control_times) / row['median_step_seconds']
                     if control_times and row['status'] == 'passed' else None)
            ratios.append(ratio)
        output[candidate] = dict(ratios_vs_best_tested_native=ratios,
                                 over_two_percent_both_orders=all(v is not None and v > 1.02 for v in ratios),
                                 convergence_proven=False, default_changed=False)
    return output


def audit(root, *, allow_partial=False):
    result_path = root / 'result.json'
    complete = result_path.exists()
    summary = json.loads(result_path.read_text()) if complete else None
    if complete:
        if (not summary['complete'] or summary['memory_budget_gib'] != 40
                or summary['case_specs'] != {name: list(spec) for name, spec in CASES.items()}
                or json.loads((root / 'status.json').read_text())['phase'] != 'finished'):
            raise ValueError('Finished matching study required')
    elif not allow_partial:
        raise ValueError('Incomplete study; explicit partial diagnostic mode required')
    rows = summary['records'] if complete else json.loads((root / 'progress.json').read_text())
    expected = [(r, n) for r, order in ((0, list(CASES)), (1, list(reversed(CASES)))) for n in order]
    if [(r['repeat'], r['case']) for r in rows] != (expected if complete else expected[:len(rows)]):
        raise ValueError('Wrong ordered study/prefix')
    signatures, runs, audited = set(), {}, []
    for row in rows:
        repeat, name = row['repeat'], row['case']
        label = f'r{repeat}_{name}'
        resource_sha = audit_resources(root / (label + '_resources.json'))
        log = (root / (label + '.log')).read_bytes()
        if hashlib.sha256(log).hexdigest() != row['log_sha256']:
            raise ValueError('Changed worker log')
        if row['status'] == 'oom':
            if row['returncode'] == 0 or not any(token in log for token in (
                    b'CUDA out of memory', b'torch.OutOfMemoryError', b'torch.cuda.OutOfMemoryError')):
                raise ValueError('Invalid capacity failure')
            audited.append(dict(case=label, status='oom', resource_sha256=resource_sha))
            continue
        directory = root / label
        raw_path = directory / 'result.json'
        if (row['status'] != 'passed' or row['returncode'] != 0
                or Path(row['source']).resolve() != raw_path.resolve()
                or hashlib.sha256(raw_path.read_bytes()).hexdigest() != row['sha256']):
            raise ValueError('Raw result provenance mismatch')
        raw, signature, manifests = audit_run(directory, CASES[name])
        if manifests != row['actual_manifests']:
            raise ValueError('Wrapper sidecar SHA changed')
        signatures.add(signature)
        snapshot = raw['manifest']['implementation_sources']
        encoded = json.dumps(snapshot['files'], sort_keys=True, separators=(',', ':')).encode()
        if (hashlib.sha256(encoded).hexdigest() != snapshot['sha256']
                or raw['manifest']['source_sha256'] != snapshot['files']['benchmarks/compare_fsdp_training.py']):
            raise ValueError('Inconsistent implementation manifest')
        if complete:
            if snapshot != summary['implementation_sources']:
                raise ValueError('Supervisor and worker implementation mismatch')
            for rank in range(4):
                sidecar = json.loads((directory / f'bounded_manifest_rank{rank}.json').read_text())
                if any(summary['helper_sources'].get(p) != h for p, h in sidecar['source_sha256'].items()):
                    raise ValueError('Supervisor and actual adapter source mismatch')
        if (raw['median_step_seconds'] != row['median_step_seconds']
                or max(v['peak_allocated_gib'] for v in raw['records']) != row['peak_allocated_gib']
                or max(v['peak_reserved_gib'] for v in raw['records']) != row['peak_reserved_gib']):
            raise ValueError('Summary differs from raw measurements')
        measured = [v for v in raw['records'] if not v['warmup']]
        audited.append(dict(case=label, status='passed', median_step_seconds=raw['median_step_seconds'],
                            measured_min_seconds=min(v['max_rank_seconds'] for v in measured),
                            measured_max_seconds=max(v['max_rank_seconds'] for v in measured),
                            peak_allocated_gib=row['peak_allocated_gib'], peak_reserved_gib=row['peak_reserved_gib'],
                            initial_hf_logits_max_abs=raw['initial_hf_parity']['max_abs'],
                            actual_adapter='BoundedScratchExchangeReduceScatter' if CASES[name][1] else 'native',
                            resource_sha256=resource_sha))
        runs[(repeat, name)] = raw
    if len(signatures) != 1:
        raise ValueError('Matching passing records required')
    decisions, trajectories = {}, {}
    if complete:
        if any((r, 'native_k32') not in runs for r in (0, 1)):
            raise ValueError('Both native K32 controls must pass')
        decisions = recompute_decisions(rows)
        if decisions != summary['decisions']:
            raise ValueError('Producer decisions differ from audited measurements')
        for repeat in (0, 1):
            for candidate in ('bounded_head_k34', 'bounded_decoder_k34'):
                for control in ('native_k32', 'native_k34'):
                    if (repeat, candidate) in runs and (repeat, control) in runs:
                        trajectories[f'r{repeat}_{candidate}_vs_{control}'] = loss_comparison(
                            runs[(repeat, candidate)], runs[(repeat, control)])
    return dict(complete=complete, partial_diagnostic_only=not complete, raw_records_audited=len(rows),
                records=audited, decisions=decisions, loss_trajectories=trajectories,
                memory_budget_gib=40, default_changed=False, convergence_proven=False,
                original_42gib_study_replaced=False,
                scope='Raw bounded-adapter40GiB study audit only; partial mode selects no winner; not arbitrary-gradient, convergence or global optimum proof')


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
