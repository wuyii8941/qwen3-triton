"""Fresh-process audit of quality stages; partial jobs never imply convergence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics

from scripts.audit_fsdp_exchange_40 import audit_resources
from scripts.campaign_fsdp_scratch_quality import audit_job
from training.fsdp_provenance import source_snapshot


PREFLIGHTS = ('tiny_native', 'tiny_bounded', 'real_native_preflight', 'real_bounded_preflight')
STAGES = PREFLIGHTS + ('native_full_protocol', 'bounded_full_protocol')


def full_comparison(native, bounded):
    if any(not r.get('complete') or not r.get('converged') or r.get('stop_reason') != 'validation_plateau'
           or not r.get('best_parameters_restored') or 'test' not in r for r in (native, bounded)):
        raise ValueError('Two complete plateau/best-restore/test jobs required')
    return dict(native_seconds=native['elapsed_seconds'], bounded_seconds=bounded['elapsed_seconds'],
        bounded_over_native_job_time=bounded['elapsed_seconds'] / native['elapsed_seconds'],
        native_updates=len(native['training']), bounded_updates=len(bounded['training']),
        native_median_recorded_step_seconds=statistics.median(v['seconds'] for v in native['training']),
        bounded_median_recorded_step_seconds=statistics.median(v['seconds'] for v in bounded['training']),
        test_loss_difference=bounded['test']['loss'] - native['test']['loss'],
        scope='Single-seed shared-hardware complete task-plateau observation; not statistical quality equivalence or isolated causal implementation speedup')


def normalize_evidence(evidence):
    result = copy.deepcopy(evidence)
    for row in result['adapter_manifests']:
        row['path'] = str(Path(row['path']).resolve())
    return result


def audit(root, *, allow_partial=False):
    path = root / 'result.json'
    complete = path.exists()
    summary = json.loads(path.read_text()) if complete else None
    if complete:
        phase = json.loads((root / 'status.json').read_text())['phase']
        if (summary['complete'] is not True or summary['default_changed'] is not False
                or phase != ('finished' if summary['full_quality_completed'] else 'preflights_passed')):
            raise ValueError('Completed quality campaign required')
    elif not allow_partial:
        raise ValueError('Incomplete campaign; explicit partial diagnostic mode required')
    rows = summary['records'] if complete else json.loads((root / 'progress.json').read_text())
    expected = STAGES if not complete or summary['full_quality_completed'] else PREFLIGHTS
    if [row['case'] for row in rows] != list(expected if complete else expected[:len(rows)]):
        raise ValueError('Wrong ordered quality stage prefix')
    selection = json.loads((root / 'selection.json').read_text())
    if (selection['selection_uses_test_loss'] is not False or selection['bounded_keep'] != 34
            or selection['bounded_scope'] != 'decoder' or selection['native_keep'] not in (32, 34)):
        raise ValueError('Wrong preselected performance policy')
    protocol_path = Path('training/fsdp_convergence_protocol_v1.json')
    protocol = json.loads(protocol_path.read_text())
    preflight_path = root / 'post_update_protocol.json'
    preflight_protocol = json.loads(preflight_path.read_text())
    expected_preflight = copy.deepcopy(protocol)
    expected_preflight['validation']['every_updates'] = 2
    if preflight_protocol != expected_preflight:
        raise ValueError('Preflight may only change validation interval')
    common = source_snapshot(Path.cwd())
    runs, records = {}, []
    for row in rows:
        name = row['case']
        bounded = name in ('tiny_bounded', 'real_bounded_preflight', 'bounded_full_protocol')
        tiny, preflight = name.startswith('tiny_'), name in PREFLIGHTS
        config_path = preflight_path if name.startswith('real_') else protocol_path
        config = preflight_protocol if name.startswith('real_') else protocol
        result_path = root / name / 'result.json'
        raw = result_path.read_bytes()
        log = (root / (name + '.log')).read_bytes()
        if (Path(row['source']).resolve() != result_path.resolve()
                or hashlib.sha256(raw).hexdigest() != row['sha256']
                or hashlib.sha256(log).hexdigest() != row['log_sha256']):
            raise ValueError('Quality raw result/log changed')
        resource_path = root / (name + '_resources.json')
        resource_sha = audit_resources(resource_path)
        resource = json.loads(resource_path.read_text())
        if (resource['min_ram_available_gib'] != 160 or resource['min_disk_free_gib'] != 16
                or any(v['mem_available_bytes'] < 160 * 2**30 or v['disk_free_bytes'] < 16 * 2**30
                       for v in resource['samples'])):
            raise ValueError('Quality RAM/disk guard not satisfied')
        result, evidence = audit_job(root / name, keep=34 if bounded else selection['native_keep'],
            bounded=bounded, protocol=config, protocol_file_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            tiny=tiny, preflight=preflight)
        if normalize_evidence(evidence) != normalize_evidence(row['evidence']) or result['source_snapshot'] != common:
            raise ValueError('Quality evidence/source differs from recorded job')
        runs[name] = result
        records.append(dict(case=name, preflight=preflight, complete=True, converged=result['converged'],
            updates=len(result['training']), validation_checks=len(result['validation']),
            validation_steps=[v['step'] for v in result['validation']],
            validation_tokens=[v['tokens'] for v in result['validation']],
            best_parameters_restored=result['best_parameters_restored'],
            peak_train_allocated_gib=max(v['max_rank_peak_allocated_gib'] for v in result['training']),
            source=str(result_path), sha256=hashlib.sha256(raw).hexdigest(), resource_sha256=resource_sha))
    comparison = None
    full = complete and summary['full_quality_completed']
    if full:
        comparison = full_comparison(runs['native_full_protocol'], runs['bounded_full_protocol'])
    if complete:
        if summary['implementation_sources'] != common or comparison != summary['full_job_comparison']:
            raise ValueError('Campaign conclusion/source differs from raw job evidence')
        for name, digest in summary['helper_sources'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError('Quality helper source changed')
        frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        if (frozen['content_sha256'] != summary['frozen_content_sha256']
                or hashlib.sha256(protocol_path.read_bytes()).hexdigest() != summary['protocol_sha256']):
            raise ValueError('Frozen input/protocol identity differs')
    return dict(complete=complete, partial_diagnostic_only=not complete, raw_records_audited=len(rows),
        records=records, all_preflights_passed=all(name in runs for name in PREFLIGHTS),
        full_quality_completed=full, full_job_comparison=comparison,
        default_changed=False, statistical_quality_equivalence_proven=False,
        checkpoints_written=False, best_parameters_exported=False,
        scope='Independent completed-stage history/source/adapter/resource audit; partial mode does not imply full quality completion')


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
