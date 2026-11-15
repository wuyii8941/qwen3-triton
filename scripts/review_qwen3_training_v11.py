"""Audit full8B training strategies; dense candidate gradients are not archived."""
from pathlib import Path
import argparse
import json
import math
import statistics
import torch

from qwen3_triton.gradient_archive import load, sha
from qwen3_training_v11_contract import audit_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.is_relative_to(Path('/home/tzh'))
    torch.set_num_threads(2)
    manifest = json.loads((root/'manifest.json').read_text())
    assert sha(root/'selection_manifest.json') == manifest['selection_manifest_sha256']
    assert sha(root/'native_feasibility.json') == manifest['native_feasibility_sha256']
    selection = json.loads((root/'selection_manifest.json').read_text())
    assert selection['minimum_reduction_each_order'] == manifest['minimum_reduction_each_order'] == .03
    assert selection['memory_cap_GiB'] == manifest['memory_cap_GiB'] == 41.875
    feasibility = json.loads((root/'native_feasibility.json').read_text())
    assert feasibility and feasibility[-1]['fits'] and not any(r['fits'] for r in feasibility[:-1])
    assert [r['strategy'] for r in feasibility] == selection['ranking'][:len(feasibility)]
    assert feasibility[-1]['strategy'] == manifest['selected_strategy']
    for row in feasibility:
        assert sha(root/(row['name']+'.gpu.json')) == row['monitor_sha256']
        assert sha(root/(row['name']+'.log')) == row['log_sha256']
        monitor = json.loads((root/(row['name']+'.gpu.json')).read_text())
        assert not monitor['foreign_process_seen'] and not monitor['observation_errors']
        if row['fits']:
            assert monitor['returncode'] == 0 and sha(root/row['name']/'result.json') == row['result_sha256']
        else:
            assert monitor['returncode'] and 'out of memory' in (root/(row['name']+'.log')).read_text().lower()
    for path, digest in manifest['sources'].items():
        assert sha(path) == digest, path
    reference = Path(manifest['reference_dir'])
    reference_result = json.loads((reference/'result.json').read_text())
    assert sha(reference/'result.json') == manifest['reference_result_sha256']
    canonical = []
    for rank in range(4):
        path = reference/f'raw_rank{rank}.pt.gz'
        assert sha(path) == reference_result['ranks'][rank]['reference_archive']['sha256']
        raw = load(path)['1']
        assert raw['ids'].shape == (1,512) and raw['logits'].shape == (1,512,151936)
        assert len(raw['gradients']) == 399
        assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in raw['gradients'].values())
        canonical.append(set(raw['gradients']))
        del raw
    plan = json.loads(Path(manifest['gemm_plan']['path']).read_text())
    assert sha(manifest['gemm_plan']['path']) == manifest['gemm_plan']['sha256']
    assert [r['name'] for r in manifest['roles']] == ['baseline_r1','hf_r1','candidate_r1','candidate_r2','hf_r2','baseline_r2']
    assert all(r['strategy'] == manifest['selected_strategy'] for r in manifest['roles'] if r['arm'] != 'baseline')
    assert all(r['strategy'] == dict(name='control',keep_full_layers=0,checkpoint_every=1,
        skip_embedding_backward_gather=False,retain_lm_head=False,gemm_plan=False)
        for r in manifest['roles'] if r['arm'] == 'baseline')
    results, timing, numerics, traces = {}, {}, {}, {}
    for role in manifest['roles']:
        name = role['name']
        monitor = json.loads((root/(name+'.gpu.json')).read_text())
        assert monitor['returncode'] == 0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
        assert monitor['samples']
        owned = set(monitor['known_owned_pids'])
        assert all(not s.get('foreign_pids') and 'observation_error' not in s
                   and all(a['pid'] in owned for a in s['apps']) for s in monitor['samples'])
        result = json.loads((root/name/'result.json').read_text())
        assert result['complete'] and not result['screen_only'] and result['experiment'] == 'training_v11'
        assert result['model'] == 'Qwen/Qwen3-8B' and result['layers'] == 36 and result['world_size'] == 4
        assert result['variant'] == role['variant'] and result['input_manifest_sha256'] == manifest['input_identity']
        assert result['input_root'] == manifest['input_root']
        assert result['math_policy'] == dict(compute='BF16', master_gradient_moments='FP32', TF32=False,
                                           allow_bf16_reduced_precision_reduction=False)
        for path, digest in result['sources'].items():
            assert sha(path) == digest, path
        expected_framework = dict(checkpoint_every=role['strategy']['checkpoint_every'],
            keep_full_layers=role['strategy']['keep_full_layers'],
            skip_embedding_backward_gather=role['strategy']['skip_embedding_backward_gather'],
            retain_lm_head=role['strategy']['retain_lm_head'], vocab_parallel=False, custom_collectives=False,
            optimizer='native_fused_AdamW', batch_per_rank=1, sequence=512)
        assert result['common_framework'] == expected_framework
        tuned = role['variant'] != 'hf' and role['strategy']['gemm_plan']
        if tuned:
            assert result['gemm_plan']['sha256'] == manifest['gemm_plan']['sha256']
        else:
            assert result['gemm_plan'] is None
        assert len(result['ranks']) == 4
        checks, samples = [], None
        for rank, row in enumerate(result['ranks']):
            assert row['rank'] == rank and row['complete'] and row['parameter_count'] == row['optimizer_state_count'] == 399
            assert row['updates'] == 27 and row['all_master_grad_moment_finite_fp32']
            assert [r['step'] for r in row['rows']] == list(range(1,28))
            assert [v['after_updates'] for v in row['full_validation']] == [0,16]
            assert all(v['blocks'] == 512 and v['tokens'] == 261632 and v['batch_per_rank'] == 4 for v in row['full_validation'])
            assert max(row['measured_training_peak'].values()) <= manifest['memory_cap_GiB']
            assert all(math.isfinite(r['global_loss']) and math.isfinite(r['wall_seconds']) and r['wall_seconds'] > 0 for r in row['rows'])
            current = [r['wall_seconds'] for r in row['rows'] if r['measured']]
            assert len(current) == 8
            if samples is None:
                samples = current
            else:
                assert samples == current
            assert [r['global_loss'] for r in row['rows']] == [r['global_loss'] for r in result['ranks'][0]['rows']]
            assert len(row['gradient_comparisons']) == 1
            comparison = row['gradient_comparisons'][0]
            strict = role['variant'] == 'hf'
            assert comparison['complete'] and comparison['strict_native_control'] == strict
            assert comparison['reference_result_sha256'] == manifest['reference_result_sha256']
            assert len(comparison['rows']) == 399 and {g['name'] for g in comparison['rows']} == canonical[rank]
            for g in comparison['rows']:
                assert all(math.isfinite(g[k]) and g[k] >= 0 for k in ('absolute_error_norm','reference_norm','relative_l2','threshold'))
                threshold = (.0001 if strict else .06)*max(g['reference_norm'],1.e-8)+(1.e-7 if strict else 1.e-6)
                assert g['threshold'] == threshold and g['passed'] == (g['absolute_error_norm'] < threshold)
            grad_passed = all(g['passed'] for g in comparison['rows'])
            forward = comparison['logits_relative_l2'] < .025 and comparison['loss_abs_error'] < .01 and (comparison['forward_bitwise_equal'] or not strict)
            assert comparison['passed'] == grad_passed and comparison['forward_gate_passed'] == forward
            error2 = sum(g['absolute_error_norm']**2 for g in comparison['rows'])
            norm2 = sum(g['reference_norm']**2 for g in comparison['rows'])
            assert comparison['global_relative_l2'] == (error2/max(norm2,1.e-16))**.5
            checks.append(dict(rank=rank, parameter_count=399, passed=grad_passed and forward,
                gradient_relative_l2=comparison['global_relative_l2'], logits_relative_l2=comparison['logits_relative_l2'],
                loss_abs_error=comparison['loss_abs_error'], forward_bitwise_equal=comparison['forward_bitwise_equal']))
        results[name] = result
        numerics[name] = checks
        timing[name] = dict(median_seconds=statistics.median(samples), samples_seconds=samples,
            peak_allocated_gib=max(r['measured_training_peak']['allocated_gib'] for r in result['ranks']),
            peak_reserved_gib=max(r['measured_training_peak']['reserved_gib'] for r in result['ranks']))
        traces[name] = audit_trace(root/name/'training_trace.json', role['variant'], role['strategy'], plan if tuned else None)
    quality = {}
    comparisons = []
    for order in (1,2):
        hf = results[f'hf_r{order}']
        candidate = results[f'candidate_r{order}']
        assert hf['common_framework'] == candidate['common_framework']
        for arm in ('baseline','hf','candidate'):
            name = f'{arm}_r{order}'
            row, ref = results[name]['ranks'][0], hf['ranks'][0]
            drift = [abs(a['global_loss']-b['global_loss']) for a,b in zip(row['rows'][:16],ref['rows'][:16])]
            validation = [abs(a['loss']-b['loss']) for a,b in zip(row['full_validation'],ref['full_validation'])]
            quality[name] = dict(passed=statistics.mean(drift)<.002 and max(drift)<.01 and max(validation)<.002,
                                 mean_training_drift=statistics.mean(drift), max_training_drift=max(drift), validation_drifts=validation)
        baseline_seconds = timing[f'baseline_r{order}']['median_seconds']
        candidate_seconds = timing[f'candidate_r{order}']['median_seconds']
        hf_seconds = timing[f'hf_r{order}']['median_seconds']
        comparisons.append(dict(order=order, baseline_seconds=baseline_seconds, candidate_seconds=candidate_seconds,
            HF_same_strategy_seconds=hf_seconds, reduction_vs_baseline=1-candidate_seconds/baseline_seconds,
            reduction_vs_HF_same_strategy=1-candidate_seconds/hf_seconds))
    gates = dict(numerics=all(r['passed'] for rows in numerics.values() for r in rows),
                 short_quality=all(r['passed'] for r in quality.values()),
                 CUDA_and_collectives=all(r['passed'] for r in traces.values()))
    performance = all(r['reduction_vs_baseline'] >= manifest['minimum_reduction_each_order'] for r in comparisons)
    report = dict(complete=True, audit_passed=True, gates=gates, all_numerical_quality_CUDA_gates_passed=all(gates.values()),
        performance_rule_passed=performance, qualified_training_improvement=all(gates.values()) and performance,
        default_changed=False, comparisons=comparisons, strategy=manifest['selected_strategy'], timing=timing,
        numerics=numerics, quality=quality, traces=traces,
        sources={str(p):sha(p) for p in [Path(__file__),root/'manifest.json',*[root/r['name']/'result.json' for r in manifest['roles']]]},
        limitation='Two orders are descriptive. Dense candidate gradient coordinates were compared at runtime, not archived. Independent replay verifies reference archives and recorded arithmetic. Sixteen quality updates do not prove convergence; inference is not measured.')
    with (root/'independent_review.json').open('x') as stream:
        stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(gates=gates, performance_rule_passed=performance, comparisons=comparisons),indent=2),flush=True)


if __name__ == '__main__':
    main()
