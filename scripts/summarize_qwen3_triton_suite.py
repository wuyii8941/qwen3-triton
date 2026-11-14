"""Summarize independently reviewed full Qwen3-8B comparisons and ablations.

This consumes completed evidence, never runs a GPU or adopts a configuration.
Two opposite-order runs provide descriptive comparisons, not a significance
test or proof that combining individually useful choices is globally optimal.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('hf', 'triton_base', 'triton_optimized', 'ablate_attention',
            'ablate_norm_rope', 'ablate_mlp', 'ablate_linear', 'ablate_loss')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module_evidence(path):
    data = json.loads(path.read_text())
    if 'report_path' in data:
        require(data.get('complete') is True and data.get('public_module_GPU_entry_passed') is True
                and data.get('resource_isolation_passed') is True, 'Module evidence incomplete')
        raw_path = Path(data['report_path'])
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        require(sha(raw_path) == data['report_sha256'], 'Module review changed')
        raw = json.loads(raw_path.read_text())
    else:
        raw = data
    require(raw.get('complete') is True and raw.get('audit_passed') is True,
            'Module independent review incomplete')
    for source, expected in raw['sources'].items():
        require(sha(Path(source)) == expected, f'Module review source changed: {source}')
    monitors = [Path(source) for source in raw['sources'] if Path(source).name == 'gpu.json']
    require(len(monitors) == 1, 'Module ownership monitor missing')
    monitor = json.loads(monitors[0].read_text())
    require(monitor['returncode'] == 0 and not monitor['foreign_process_seen']
            and monitor['observation_errors'] == 0, 'Module resource isolation failed')
    summary = {}
    require(len(raw['summary']) == 96, 'Module case summary incomplete')
    for kind in ('attention', 'norm_rope', 'mlp', 'lm_head', 'loss'):
        rows = [r for r in raw['summary'] if r['module'] == kind and r['variant'] == 'triton_optimized']
        require(len(rows) == (6 if kind in ('attention', 'norm_rope', 'mlp') else 2),
                f'Module order/layer coverage incomplete: {kind}')
        summary[kind] = dict(cases=len(rows),
                            median_sum_of_separate_fwd_bwd_medians_ms=statistics.median(
                                r['sum_of_separate_medians_ms'] for r in rows),
                            median_latency_reduction_vs_HF=statistics.median(
                                r['fwd_bwd_reduction_vs_HF'] for r in rows),
                            median_latency_reduction_vs_Triton_base=statistics.median(
                                r['fwd_bwd_reduction_vs_Triton_base'] for r in rows))
    return dict(raw, isolated_module_summary=summary)


def summarize(suite_path, modules_path, recovery_paths=()):
    review = json.loads(suite_path.read_text())
    modules = module_evidence(modules_path)
    for key in ('complete', 'audit_passed', 'trainability_and_state_checks_passed',
                'strict_BF16_numeric_gate_passed', 'all_short_quality_contracts_passed',
                'CUDA_training_trace_coverage_passed'):
        require(review.get(key) is True, f'Suite has not passed {key}')
    require(review['model'] == modules['model'] == 'Qwen/Qwen3-8B', 'Official8B required')
    require(review['layers'] == 36 and review['world_size'] == 4, 'Full36layer four-rank suite required')
    require(modules['case_count'] == 96 and modules['orders'] == 2
            and modules['timing_samples_per_part'] == 20, 'Module protocol mismatch')
    role_map = {(r['variant'], r['order']): r['name'] for r in review['roles']}
    require(len(review['roles']) == len(role_map) == 16
            and set(role_map) == {(v, order) for v in VARIANTS for order in (1, 2)},
            'Need all eight variants in both orders')
    for path, expected in review['sources'].items():
        require(sha(Path(path)) == expected, f'Review source changed: {path}')
    discovered = {p.resolve() for p in suite_path.parent.glob('recovery_*/manifest.json')}
    provided = {p.resolve() for p in recovery_paths}
    require(discovered == provided, 'Explicitly provide every cohort recovery manifest; interruptions cannot be hidden')
    recovery_notes = []
    for path in sorted(provided):
        recovery = json.loads(path.read_text())
        require(recovery['interrupted_cohort'] is True and recovery['training_or_protocol_changed'] is False
                and recovery['original_relative_order_preserved'] is True, 'Invalid recovery policy')
        require(sha(suite_path.parent/'manifest.json') == recovery['original_manifest_sha256'],
                'Original cohort manifest changed during recovery')
        count = len(recovery['retained_roles'])
        require(0 < count < 16 and recovery['retained_roles'] == review['roles'][:count]
                and recovery['remaining_roles'] == review['roles'][count:], 'Recovery role order mismatch')
        for source, expected in recovery['original_clean_sources'].items():
            require(sha(Path(source)) == expected, f'Retained result changed: {source}')
        rejected = Path(recovery['rejected_attempt_directory'])
        for relative, expected in recovery['rejected_attempt_hashes'].items():
            require(sha(rejected/relative) == expected, f'Rejected evidence changed: {relative}')
        recovery_notes.append(dict(manifest=str(path), manifest_sha256=sha(path),
                                   retained_clean_roles=[r['name'] for r in recovery['retained_roles']],
                                   subsequent_roles=[r['name'] for r in recovery['remaining_roles']],
                                   rejected_attempt_directory=str(rejected), reason=recovery['reason'],
                                   training_or_protocol_changed=False, original_relative_order_preserved=True))

    comparisons = []
    for variant in VARIANTS:
        orders = []
        for order in (1, 2):
            name = role_map[variant, order]
            row = review['timing'][name]
            require(row['variant'] == variant and row['order'] == order, 'Timing identity mismatch')
            samples = row['seconds']
            require(len(samples) == 8 and all(math.isfinite(x) and x > 0 for x in samples),
                    f'Invalid whole-step samples: {name}')
            require(row['median_seconds'] == statistics.median(samples), f'Median mismatch: {name}')
            hf = review['timing'][role_map['hf', order]]['median_seconds']
            base = review['timing'][role_map['triton_base', order]]['median_seconds']
            for key in ('measured_peak_allocated_gib', 'measured_peak_reserved_gib'):
                require(math.isfinite(row[key]) and 0 < row[key] <= 41.875, f'Invalid memory: {name}')
            quality = review['quality'][name]
            require(quality['short_quality_contract_passed'] is True, f'Quality failure: {name}')
            require([p['after_updates'] for p in quality['full_validation']] == [0, 16],
                    f'Validation points missing: {name}')
            gradients = review['gradient_checks'][name]
            # A freshly exported native r1 reference has no comparison to itself.
            require(len(gradients) == (0 if name == 'hf_reference_r1' else 4)
                    or (name == 'hf_reference_r1' and len(gradients) == 4),
                    f'Gradient-rank evidence missing: {name}')
            require(all(g['all_parameter_gradient_gate_passed'] and g['forward_gate_passed']
                        and g['gradient_failure_count'] == 0 for g in gradients), f'Gradient failure: {name}')
            for g in gradients:
                require(all(math.isfinite(g[key]) and g[key] >= 0 for key in
                            ('global_gradient_relative_l2', 'logits_relative_l2', 'loss_abs_error')),
                        f'Numeric error metrics missing/nonfinite: {name}')
            trace = review['training_CUDA_traces'][name]
            if variant == 'hf':
                require(trace.get('reference') is True and bool(trace['native_Flash_kernels']),
                        f'Native Flash missing: {name}')
            else:
                require(trace.get('coverage_passed') is True and not trace['forbidden_library_kernels']
                        and not trace['unclassified_kernels'] and not trace['count_discrepancies'],
                        f'Triton coverage failure: {name}')
            orders.append(dict(order=order, median_step_seconds=row['median_seconds'],
                               step_seconds=samples, latency_reduction_vs_HF=1-row['median_seconds']/hf,
                               latency_reduction_vs_Triton_base=1-row['median_seconds']/base,
                               peak_allocated_gib=row['measured_peak_allocated_gib'],
                               peak_reserved_gib=row['measured_peak_reserved_gib'],
                               validation_loss_after16=quality['full_validation'][1]['loss'],
                               validation_loss_drift_after16=quality['full_validation'][1]['absolute_drift']))
        hf_ratio = math.sqrt(math.prod(1-r['latency_reduction_vs_HF'] for r in orders))
        base_ratio = math.sqrt(math.prod(1-r['latency_reduction_vs_Triton_base'] for r in orders))
        comparisons.append(dict(variant=variant, orders=orders,
                                geometric_mean_latency_reduction_vs_HF=1-hf_ratio,
                                geometric_mean_latency_reduction_vs_Triton_base=1-base_ratio))
    ablations = []
    for variant in VARIANTS[3:]:
        increases = [review['timing'][role_map[variant, order]]['median_seconds'] /
                     review['timing'][role_map['triton_optimized', order]]['median_seconds'] - 1
                     for order in (1, 2)]
        interpretation = ('benefit_in_both_orders' if min(increases) > 0 else
                          'no_benefit_in_both_orders' if max(increases) <= 0 else 'mixed_between_orders')
        ablations.append(dict(disabled=variant.removeprefix('ablate_'),
                              ablation_step_increase_fraction_by_order=increases,
                              interpretation=interpretation,
                              individual_ablation_does_not_establish_optimal_combination=True))
    return dict(complete=True, model='Qwen/Qwen3-8B', layers=36, world_size=4,
                comparisons=comparisons, ablations=ablations,
                isolated_module_summary=modules['isolated_module_summary'],
                numerical_checks=review['gradient_checks'],
                short_training_quality=review['quality'],
                training_CUDA_traces=review['training_CUDA_traces'],
                parameter_tensors_per_rank=399, actual_updates_per_role=27,
                candidate_gradient_gate='L2_error < 0.06 * max(reference_L2, 1e-8) + 1e-6 per parameter shard',
                native_gradient_control_gate='L2_error < 0.0001 * max(reference_L2, 1e-8) + 1e-7; bitwise forward',
                dense_candidate_gradient_replay=False,
                gradient_evidence_scope='All-coordinate runtime comparisons on the first update; independent CPU review replays native archives and candidate gate arithmetic. Dense candidate vectors are not archived.',
                paired_quality_and_coverage_passed=True, performance_adopted=False,
                convergence_or_inference_speedup_claimed=False,
                interrupted_cohort=bool(recovery_notes), cohort_recovery=recovery_notes,
                scope=__doc__, inputs={str(p): sha(p) for p in (suite_path, modules_path, *sorted(provided))},
                summarizer_sha256=sha(Path(__file__)))


def markdown(report):
    lines = ['# Official Qwen3-8B Triton comparison', '',
             'Four A6000 GPUs; full 36-layer model. Opposite-order timings are descriptive.',
             'All arms share native FSDP2, full-layer checkpointing and fused AdamW.', '',
             '| Variant | Step s, order 1 / 2 | Latency reduction vs HF, order 1 / 2 | Peak allocated GiB, order 1 / 2 |',
             '| --- | ---: | ---: | ---: |']
    if report['interrupted_cohort']:
        lines[4:4] = ['Cohort interrupted by a resource-isolation failure: rejected measurements were preserved',
                      'and excluded. Clean roles were retained and the remainder rerun with unchanged source/policy.',
                      'The JSON binds recovery receipts and rejected-evidence hashes.', '']
    for row in report['comparisons']:
        orders = row['orders']
        times = ' / '.join(f"{r['median_step_seconds']:.4f}" for r in orders)
        reductions = ' / '.join(f"{r['latency_reduction_vs_HF']:+.2%}" for r in orders)
        memory = ' / '.join(f"{r['peak_allocated_gib']:.3f}" for r in orders)
        lines.append(f"| {row['variant']} | {times} | {reductions} | {memory} |")
    lines.extend(['', 'Positive ablation increases indicate a benefit from the enabled choice.', '',
                  '| Disabled optimization | Step increase, order 1 / 2 | Observation |',
                  '| --- | ---: | --- |'])
    for row in report['ablations']:
        delta = ' / '.join(f'{v:+.2%}' for v in row['ablation_step_increase_fraction_by_order'])
        lines.append(f"| {row['disabled']} | {delta} | {row['interpretation']} |")
    lines.extend(['', 'Individual ablations do not prove the best combined configuration.',
                  'Module timings exclude FSDP/NCCL/AdamW and cannot substitute for whole-step measurements.', '',
                  '| Isolated module | Median latency reduction vs HF |', '| --- | ---: |'])
    for name, row in report['isolated_module_summary'].items():
        lines.append(f"| {name} | {row['median_latency_reduction_vs_HF']:+.2%} |")
    lines.extend(['', '| Role | Max rank gradient relative L2 | Max logits relative L2 | Max loss absolute error |',
                  '| --- | ---: | ---: | ---: |'])
    for name, rows in report['numerical_checks'].items():
        if not rows:
            lines.append(f'| {name} | Native reference; no self-comparison | — | — |')
            continue
        gradient = max(r['global_gradient_relative_l2'] for r in rows)
        logits = max(r['logits_relative_l2'] for r in rows)
        loss = max(r['loss_abs_error'] for r in rows)
        lines.append(f'| {name} | {gradient:.6g} | {logits:.6g} | {loss:.6g} |')
    lines.extend(['', 'Each candidate parameter shard passes `L2_error < 0.06 * max(reference_L2, 1e-8) + 1e-6`.',
                  'The table shows rank-wide gradient norms; the per-parameter gate is checked separately.',
                  'All-coordinate candidate checks run on the first update; dense candidate gradients are not archived.',
                  'The JSON preserves rank-level numeric checks, all short-quality measurements and CUDA kernel coverage.'])
    lines.extend(['', 'All paired short-quality and CUDA-coverage gates passed. This is not a convergence',
                  'or inference result. No optimization configuration is automatically adopted.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True, help='Completed independent_review.json')
    parser.add_argument('--modules', type=Path, required=True, help='Module independent_review.json or curated V9 report')
    parser.add_argument('--recovery', type=Path, action='append', default=[],
                        help='Every recovery manifest for an interrupted cohort; repeat for multiple recoveries')
    parser.add_argument('--output', type=Path, required=True, help='New directory for JSON and Markdown')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(Path('/home/tzh')):
        parser.error('Output under /home/tzh is forbidden')
    report = summarize(args.suite.resolve(), args.modules.resolve(), args.recovery)
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'summary.md').write_text(markdown(report))
    print(json.dumps(dict(complete=True, output=str(output), performance_adopted=False)))


if __name__ == '__main__':
    main()
