"""Fresh full plateau comparison with matched conservative expandable caps."""
import argparse
import hashlib
import json
from pathlib import Path
import signal

from benchmarks.fsdp_exchange_quality_expandable import ALLOCATOR, EFFECTIVE_CAP_GIB
from scripts.audit_fsdp_exchange_confirmation import audit as audit_performance, audit_launch
from scripts.audit_fsdp_exchange_expandable_preflights import audit as audit_preflights
from scripts.audit_fsdp_scratch_quality import full_comparison
from scripts.campaign_fsdp_exchange_quality import audit_full, selection, helpers as base_helpers
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.preflight_fsdp_exchange_expandable import (
    audit_allocator, command, environment, resources, hashes as preflight_helpers)
from training.fsdp_provenance import source_snapshot


def helpers():
    names = ('scripts/campaign_fsdp_exchange_expandable.py', 'scripts/audit_fsdp_exchange_expandable.py',
             'scripts/audit_fsdp_exchange_expandable_preflights.py')
    return dict(base_helpers(), **preflight_helpers(), **{
        n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names})


def chosen_policy(rows):
    return dict(selection(rows), allocator=ALLOCATOR, effective_allocator_cap_gib=EFFECTIVE_CAP_GIB,
                audited_reserved_upper_bound_gib=42.)


def audit(root, *, allow_partial=False):
    final = root / 'result.json'
    complete = final.exists()
    if not complete and not allow_partial:
        raise ValueError('Both complete plateau jobs required')
    summary = json.loads(final.read_text()) if complete else None
    rows = summary['records'] if complete else json.loads((root / 'progress.json').read_text())
    if ([r['case'] for r in rows] != ['native', 'decoder'][:len(rows)] or len(rows) > 2
            or (complete and len(rows) != 2)):
        raise ValueError('Wrong ordered full job coverage')
    original = Path('training/fsdp_convergence_protocol_v1.json')
    performance = json.loads(Path('artifacts/fsdp_exchange_model_confirmation_v2/result.json').read_text())
    if json.loads((root / 'selection.json').read_text()) != chosen_policy(performance['records']):
        raise ValueError('Unintended allocator/policy or quality-based selection')
    runs, evidence = {}, []
    for row in rows:
        name = row['case']
        path = root / name / 'result.json'
        if (Path(row['source']).resolve() != path.resolve()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']
                or hashlib.sha256((root / f'{name}.log').read_bytes()).hexdigest() != row['log_sha256']):
            raise ValueError('Full job raw data/log changed')
        result, history = audit_full(root / name, name, original)
        contexts = audit_allocator(root / name)
        if history != row['evidence'] or contexts != row['allocator_sha256']:
            raise ValueError('Full history/adapter/allocator evidence mismatch')
        resource_sha = audit_launch(root, name,
            dict(allocator_budget_gib=42., free_margin_gib=2.5, idle_confirmations=3))
        launch = json.loads((root / f'{name}_resources.json').read_text())
        if any(s['mem_available_bytes'] < 160 * 2**30 or s['disk_free_bytes'] < 16 * 2**30 for s in launch['samples']):
            raise ValueError('RAM/disk launch guard violated')
        runs[name] = result
        evidence.append(dict(case=name, sha256=row['sha256'], resource_sha256=resource_sha,
                             evidence=history, allocator_sha256=contexts))
    comparison = full_comparison(runs['native'], runs['decoder']) if complete else None
    if complete:
        if (summary['complete'] is not True or summary['full_quality_completed'] is not True
                or summary['comparison'] != comparison or summary['helper_sources'] != helpers()
                or summary['implementation_sources'] != source_snapshot(Path.cwd())
                or summary['protocol_sha256'] != hashlib.sha256(original.read_bytes()).hexdigest()
                or summary['allocator'] != ALLOCATOR or summary['effective_allocator_cap_gib'] != EFFECTIVE_CAP_GIB
                or summary['audited_reserved_upper_bound_gib'] != 42.
                or summary['defaults_changed'] is not False or summary['checkpoints_written'] is not False
                or summary['best_parameters_exported'] is not False
                or json.loads((root / 'status.json').read_text())['phase'] != 'finished'):
            raise ValueError('Full summary/status/source/configuration mismatch')
        frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        if frozen['content_sha256'] != summary['frozen_content_sha256']:
            raise ValueError('Frozen input identity mismatch')
    return dict(complete=complete, partial_diagnostic_only=not complete, raw_jobs_audited=len(rows),
        evidence=evidence, comparison=comparison, full_quality_completed=complete,
        allocator=ALLOCATOR, effective_allocator_cap_gib=EFFECTIVE_CAP_GIB, audited_reserved_upper_bound_gib=42.,
        defaults_changed=False, checkpoints_written=False, best_parameters_exported=False,
        statistical_quality_equivalence_proven=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def status(phase, **kw):
        (args.output / 'status.json').write_text(json.dumps(dict(phase=phase, **kw), indent=2))
        print(phase, kw, flush=True)
    def interrupt(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    common, before = source_snapshot(Path.cwd()), helpers()
    protocol = Path('training/fsdp_convergence_protocol_v1.json')
    protocol_raw = protocol.read_bytes()
    def unchanged():
        if source_snapshot(Path.cwd()) != common or helpers() != before or protocol.read_bytes() != protocol_raw:
            raise ValueError('Full comparison implementation/protocol changed')
    try:
        status('auditing_prior_gates')
        preflight = audit_preflights(Path('artifacts/fsdp_exchange_expandable_preflights_v2'))
        performance_root = Path('artifacts/fsdp_exchange_model_confirmation_v2')
        performance = audit_performance(performance_root)
        raw_performance = json.loads((performance_root / 'result.json').read_text())
        if (not preflight['all_preflights_passed'] or preflight['effective_allocator_cap_gib'] != EFFECTIVE_CAP_GIB
                or not performance['decisions']['torch_decoder_auto_k32']['over_two_percent_both_orders']
                or raw_performance['implementation_sources'] != common):
            raise ValueError('Matching-source complete preflights/performance required')
        (args.output / 'selection.json').write_text(json.dumps(chosen_policy(raw_performance['records']), indent=2))
        frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        verify_frozen(frozen)
        rows, runs = [], {}
        for mode in ('native', 'decoder'):
            status('waiting_for_idle_gpus', case=mode)
            resources(args.output, mode)
            unchanged()
            status('running_full_quality', case=mode)
            log = args.output / f'{mode}.log'
            timeout = json.loads(protocol_raw)['safety']['max_hours_per_backend'] * 3600 + 1800
            if run_owned(command(args.output / mode, mode, protocol, preflight=False), log, environment(), timeout=timeout):
                raise RuntimeError(f'{mode} failed; no fallback')
            unchanged()
            result, evidence = audit_full(args.output / mode, mode, protocol)
            contexts = audit_allocator(args.output / mode)
            runs[mode] = result
            rows.append(dict(case=mode, source=str(args.output / mode / 'result.json'),
                sha256=hashlib.sha256((args.output / mode / 'result.json').read_bytes()).hexdigest(),
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), evidence=evidence, allocator_sha256=contexts))
            (args.output / 'progress.json').write_text(json.dumps(rows, indent=2))
        verify_frozen(frozen)
        unchanged()
        (args.output / 'result.json').write_text(json.dumps(dict(complete=True, full_quality_completed=True,
            records=rows, comparison=full_comparison(runs['native'], runs['decoder']),
            implementation_sources=common, helper_sources=before,
            protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(), frozen_content_sha256=frozen['content_sha256'],
            allocator=ALLOCATOR, effective_allocator_cap_gib=EFFECTIVE_CAP_GIB, audited_reserved_upper_bound_gib=42.,
            defaults_changed=False, checkpoints_written=False, best_parameters_exported=False), indent=2))
        status('finished')
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
