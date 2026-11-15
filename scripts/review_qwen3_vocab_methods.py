"""Separate accepted Triton methods from a rejected auxiliary HF port.

Preserves the complete original cohort report, including every failure. Each
recommended method and both matched-order controls must pass all original
gradient, quality, memory, CUDA and >=3% performance gates. No tolerance changes.
"""
from pathlib import Path
import argparse,json,math,statistics,sys
from qwen3_triton.gradient_archive import sha
from qwen3_vocab_v12_contract import audit_trace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();assert out.is_relative_to(Path('/data1/tzh'))
    manifest=json.loads((out/'manifest.json').read_text())
    cohort=json.loads((out/'independent_review.json').read_text())
    resources=json.loads((out/'resource_review.json').read_text())
    assert cohort['complete'] and cohort['audit_passed'] and resources['complete']
    assert resources['first_clean_attempt_policy_passed'] and resources['no_timing_based_attempt_selection']
    assert len(resources['accepted'])==len(manifest['roles'])==10
    assert resources['manifest_sha256']==sha(out/'manifest.json') and resources['attempt_history_sha256']==sha(out/'attempts.json')
    for record in [manifest,cohort]:
        for path,digest in record['sources'].items():assert sha(path)==digest,path
    assert manifest['memory_cap_GiB']==41.875 and manifest['minimum_reduction_each_order']==.03
    assert manifest['checkpoint_every']==1 and manifest['fused_dw']
    assert cohort['qualified_training_improvement']==(all(cohort['gates'].values()) and cohort['performance_rule_passed'])
    plan=json.loads(Path(manifest['gemm_plan']['path']).read_text())
    data={r['name']:json.loads((out/r['name']/'result.json').read_text()) for r in manifest['roles']}
    role_gates={};failed_gradients={};timing={};traces={}
    for role in manifest['roles']:
        name=role['name'];a=data[name];strict=role['arm']=='native'
        assert a['complete'] and not a['screen_only'] and a['model']=='Qwen/Qwen3-8B' and a['layers']==36 and a['world_size']==4
        assert a['input_manifest_sha256']==manifest['input_identity'] and len(a['ranks'])==4
        numerics=True;bad=[];samples=None
        for rank,row in enumerate(a['ranks']):
            assert row['rank']==rank and row['updates']==27 and row['parameter_count']==row['optimizer_state_count']==399
            assert row['all_master_grad_moment_finite_fp32'] and max(row['measured_training_peak'].values())<=41.875
            c=row['gradient_comparisons'][0];assert len(c['rows'])==399 and c['strict_native_control']==strict
            canonical={g['name'] for g in data[f"native_r{role['order']}"]['ranks'][rank]['gradient_comparisons'][0]['rows']}
            assert {g['name'] for g in c['rows']}==canonical
            for g in c['rows']:
                threshold=(.0001 if strict else .06)*max(g['reference_norm'],1e-8)+(1e-7 if strict else 1e-6)
                assert g['threshold']==threshold and g['passed']==(g['absolute_error_norm']<threshold)
                if not g['passed']:bad.append(dict(rank=rank,**g))
            forward=c['logits_relative_l2']<.025 and c['loss_abs_error']<.01 and (c['forward_bitwise_equal'] or not strict)
            numerics=numerics and all(g['passed'] for g in c['rows']) and forward
            times=[r['wall_seconds'] for r in row['rows'] if r['measured']];assert len(times)==8
            if samples is None:samples=times
            else:assert samples==times
            assert [v['after_updates'] for v in row['full_validation']]==[0,16]
            assert all(v['blocks']==512 and v['tokens']==261632 for v in row['full_validation'])
        row=a['ranks'][0];ref=data[f"native_r{role['order']}"]['ranks'][0]
        drift=[abs(x['global_loss']-y['global_loss']) for x,y in zip(row['rows'][:16],ref['rows'][:16])]
        val=[abs(x['loss']-y['loss']) for x,y in zip(row['full_validation'],ref['full_validation'])]
        quality=statistics.mean(drift)<.002 and max(drift)<.01 and max(val)<.002
        trace=audit_trace(out/name/'training_trace.json',role['variant'],role['vocab_ce'],plan if role['variant']=='triton_optimized' else None,
            role['checkpoint_every'],fused_dw=role['fused_dw'],dw_config=(64,128,32,8,8,3) if role['fused_dw'] else None)
        assert trace['trace_sha256']==cohort['traces'][name]['trace_sha256']
        role_gates[name]=dict(numerics=numerics,short_quality=quality,actual_CUDA=trace['passed'],memory_and_FP32_states=True,
            mean_loss_drift=statistics.mean(drift),max_loss_drift=max(drift),validation_drifts=val)
        failed_gradients[name]=bad;traces[name]=trace
        timing[name]=dict(median_seconds=statistics.median(samples),samples_seconds=samples,
            allocated_GiB=max(r['measured_training_peak']['allocated_gib'] for r in a['ranks']),
            reserved_GiB=max(r['measured_training_peak']['reserved_gib'] for r in a['ranks']))
        assert timing[name]==cohort['timing'][name]
    valid=lambda name:all(role_gates[name][k] for k in ['numerics','short_quality','actual_CUDA','memory_and_FP32_states'])
    assert cohort['gates']['numerics']==all(g['numerics'] for g in role_gates.values())
    assert cohort['gates']['short_quality']==all(g['short_quality'] for g in role_gates.values())
    assert cohort['gates']['actual_CUDA_and_collectives']==all(g['actual_CUDA'] for g in role_gates.values())
    controls=all(valid(f'{arm}_r{o}') for arm in ['native','triton_control'] for o in [1,2])
    methods={}
    for arm in ['head_unfused','candidate','HF_custom_head']:
        own_valid=all(valid(f'{arm}_r{o}') for o in [1,2])
        fusion_gate=arm!='candidate' or cohort['fusion_recorded_trajectory_equal'] is True
        comparisons=[]
        for order in [1,2]:
            t=timing[f'{arm}_r{order}']['median_seconds'];hf=timing[f'native_r{order}']['median_seconds'];old=timing[f'triton_control_r{order}']['median_seconds']
            comparisons.append(dict(order=order,seconds=t,native_HF_seconds=hf,triton_V11_seconds=old,
                reduction_vs_native_HF=1-t/hf,reduction_vs_triton_V11=1-t/old))
        performance=all(c['reduction_vs_native_HF']>=.03 and c['reduction_vs_triton_V11']>=.03 for c in comparisons)
        methods[arm]=dict(controls_qualified=controls,own_all_original_gates_passed=own_valid,
            fusion_recorded_trajectory_gate=fusion_gate,performance_rule_passed=performance if controls and own_valid and fusion_gate else None,
            qualified=controls and own_valid and fusion_gate and performance,
            performance_eligible=controls and own_valid and fusion_gate,
            comparisons=comparisons if controls and own_valid and fusion_gate else None,
            failed_gradients={f'r{o}':failed_gradients[f'{arm}_r{o}'] for o in [1,2]})
    qualified=[a for a in ['candidate','head_unfused'] if methods[a]['qualified']]
    recommended=None
    if qualified:
        if 'candidate' not in qualified:recommended='head_unfused'
        elif 'head_unfused' not in qualified:recommended='candidate'
        elif all(timing[f'candidate_r{o}']['median_seconds']<=timing[f'head_unfused_r{o}']['median_seconds'] for o in [1,2]):recommended='candidate'
        else:recommended='head_unfused'
    base=out.parent;policy=base/'per_method_adoption_policy_001.json'
    assert json.loads(policy.read_text())['thresholds_unchanged']
    report=dict(complete=True,audit_passed=True,all_original_thresholds_unchanged=True,
        whole_cohort_all_profiles_passed=cohort['qualified_training_improvement'],original_cohort_review_sha256=sha(out/'independent_review.json'),
        native_HF_and_V11_controls_qualified=controls,methods=methods,role_gates=role_gates,
        recommended_arm=recommended,recommended_method_qualified=recommended is not None,
        recommended_fused_DW=recommended=='candidate',timing=timing,
        default_changed=False,excluded_from_performance_claims=[a for a,v in methods.items() if not v['performance_eligible']],
        sources={str(p):sha(p) for p in [Path(__file__),policy,out/'manifest.json',out/'independent_review.json',out/'resource_review.json',*[out/r['name']/'result.json' for r in manifest['roles']]]},
        scope='Original cohort report is preserved, including rejected auxiliary ports. Only individually qualified methods and qualified matched-order controls support performance claims. No gradient tolerance or quality/memory/performance threshold is relaxed. Not a convergence or inference claim.')
    with (out/'per_method_review.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(controls_qualified=controls,recommended_arm=recommended,method_qualification={a:v['qualified'] for a,v in methods.items()},excluded_from_performance_claims=report['excluded_from_performance_claims']),indent=2))


if __name__=='__main__':main()
