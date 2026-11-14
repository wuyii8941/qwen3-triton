"""Independent audit of paired complete8B training/quality/ablation evidence.

Numerical failures are data, never assertion-aborted or silently adopted.
Candidate dense gradients were checked at runtime against archived nativeHF
vectors; independent CPU replay verifies the reference archive and recorded
gate arithmetic, not unavailable dense candidate vectors.
"""
from pathlib import Path
from collections import Counter
import argparse,json,math,statistics
import torch
from qwen3_triton.gradient_archive import sha,load
D=None;REFERENCE=None
def trace_report(path,variant):
    data=json.loads(path.read_text());counts=Counter(e['name'] for e in data['traceEvents'] if e.get('cat')=='kernel')
    if variant=='hf':
        flash={name:count for name,count in counts.items() if 'flash' in name.lower()}
        assert flash,'Native reference must actually execute Flash kernels'
        return dict(reference=True,native_Flash_kernels=flash,trace_sha256=sha(path))
    fused_norm=variant not in ('triton_base','ablate_norm_rope');fused_mlp=variant not in ('triton_base','ablate_mlp')
    expected={'_gemm':867 if fused_mlp else 1011,'_native_precision_forward':72,'_native_split_dq':36,'_dq_split_reduce':36,'_native_precision_dkdv':36,'_native_order_delta':36,'_attention_kv_head_reduce_kernel':36,'_hf_norm_forward':145 if fused_norm else 289,'_hf_swiglu_backward':36,'_reduce_rows':290,'_lookup':1,'_zero':1,'_positions':1,'_tables':1,'_partials':1,'_rows':1,'_mean':1,'_backward':75 if fused_norm else 147,'_add':144 if fused_mlp else 108}
    if fused_norm:expected.update(_hf_norm_rope_forward=144,_norm_rope_backward=72)
    else:expected['_apply']=216
    if fused_mlp:expected['_gate_up']=72
    else:expected['_hf_swiglu_forward']=72
    discrepancies={name:dict(expected=count,observed=counts[name]) for name,count in expected.items() if counts[name]!=count}
    forbidden={name:count for name,count in counts.items() if name not in expected and any(s in name.lower() for s in ('cublas','cutlass','sgemm','hgemm','gemm','flash','xmma'))}
    framework={};unknown={}
    for name,count in counts.items():
        if name in expected:continue
        if any(token in name for token in ('FillFunctor','Copy<','split_with_sizes_copy_out_contiguous_no_cast_kernel','direct_copy_kernel_cuda','chunk_cat_cuda_kernel','CUDAFunctor_add','ncclDevKernel_')):framework[name]=count
        else:unknown[name]=count
    return dict(reference=False,coverage_passed=not discrepancies and not forbidden and not unknown,observed_expected_kernels={name:counts[name] for name in expected},count_discrepancies=discrepancies,forbidden_library_kernels=forbidden,unclassified_kernels=unknown,framework_kernels=framework,trace_sha256=sha(path))
def main():
    global D,REFERENCE
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();D=args.output.resolve()
    if D.is_relative_to(Path('/home/tzh')):parser.error('Output under /home/tzh is forbidden')
    torch.set_num_threads(2);manifest=json.loads((D/'manifest.json').read_text())
    REFERENCE=Path(manifest.get('reference_dir',str(D/'hf_reference_r1'))).resolve()
    for path,h in manifest['sources'].items():assert sha(path)==h
    reference_result=json.loads((REFERENCE/'result.json').read_text())
    if 'reference_result_sha256' in manifest:assert sha(REFERENCE/'result.json')==manifest['reference_result_sha256']
    for rank in range(4):
        archive=REFERENCE/f'raw_rank{rank}.pt.gz'
        assert sha(archive)==reference_result['ranks'][rank]['reference_archive']['sha256']
        raw=load(archive)['1']
        assert raw['ids'].shape==(1,512) and len(raw['gradients'])==399 and raw['logits'].shape==(1,512,151936)
        assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in raw['gradients'].values())
        del raw
    assert len(manifest['roles'])==16 and manifest['quality_steps']==16 and manifest['warmup_steps']==3 and manifest['measure_steps']==8
    results={};timing={};traces={};gradient_checks={}
    for role in manifest['roles']:
        name=role['name'];variant=role['variant'];monitor=json.loads((D/(name+'.gpu.json')).read_text());result=json.loads((D/name/'result.json').read_text())
        assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and monitor['observation_errors']==0
        assert result['complete'] and result['model']=='Qwen/Qwen3-8B' and result['layers']==36 and result['world_size']==4 and result['variant']==variant
        assert result['performance_adopted'] is False and result['numerical_equivalence_qualified'] is False
        assert result['math_policy']==dict(compute='BF16',master_gradient_moments='FP32',TF32=False,allow_bf16_reduced_precision_reduction=False)
        for path,h in result['sources'].items():assert sha(path)==h
        results[name]=result;gradients=[]
        for rank,row in enumerate(result['ranks']):
            assert row['rank']==rank and row['complete'] and row['parameter_count']==row['optimizer_state_count']==399 and row['updates']==27 and row['all_master_grad_moment_finite_fp32']
            assert [r['step'] for r in row['rows']]==list(range(1,28)) and sum(r['measured'] for r in row['rows'])==8
            assert [v['after_updates'] for v in row['full_validation']]==[0,16]
            for point in row['full_validation']:assert point['blocks']==512 and point['tokens']==261632 and point['batch_per_rank']==4
            assert max(row['measured_training_peak'].values())<=41.875
            if name=='hf_reference_r1' and not row['gradient_comparisons']:
                assert REFERENCE==(D/name).resolve()
                assert sha(REFERENCE/f'raw_rank{rank}.pt.gz')==row['reference_archive']['sha256']
                continue
            assert len(row['gradient_comparisons'])==1;comparison=row['gradient_comparisons'][0]
            assert comparison['complete'] and len(comparison['rows'])==399 and comparison['reference_result_sha256']==sha(REFERENCE/'result.json')
            strict_native=variant=='hf'
            assert comparison['strict_native_control']==strict_native
            for g in comparison['rows']:
                threshold=(.0001 if strict_native else .06)*max(g['reference_norm'],1e-8)+(1e-7 if strict_native else 1e-6)
                assert g['threshold']==threshold and g['passed']==(g['absolute_error_norm']<threshold)
            assert comparison['passed']==all(g['passed'] for g in comparison['rows'])
            forward=comparison['logits_relative_l2']<.025 and comparison['loss_abs_error']<.01 and (comparison['forward_bitwise_equal'] or not strict_native)
            assert comparison['forward_gate_passed']==forward
            gradients.append(dict(rank=rank,gradient_failure_count=sum(not g['passed'] for g in comparison['rows']),all_parameter_gradient_gate_passed=comparison['passed'],global_gradient_relative_l2=comparison['global_relative_l2'],logits_relative_l2=comparison['logits_relative_l2'],loss_abs_error=comparison['loss_abs_error'],forward_gate_passed=forward,failures=[g for g in comparison['rows'] if not g['passed']]))

        samples=[r['wall_seconds'] for r in result['ranks'][0]['rows'] if r['measured']]
        for rank in result['ranks'][1:]:
            assert samples==[r['wall_seconds'] for r in rank['rows'] if r['measured']]
            assert [r['global_loss'] for r in rank['rows']]==[r['global_loss'] for r in result['ranks'][0]['rows']]
        timing[name]=dict(variant=variant,order=role['order'],seconds=samples,median_seconds=statistics.median(samples),measured_peak_allocated_gib=max(r['measured_training_peak']['allocated_gib'] for r in result['ranks']),measured_peak_reserved_gib=max(r['measured_training_peak']['reserved_gib'] for r in result['ranks']))
        gradient_checks[name]=gradients;traces[name]=trace_report(D/name/'training_trace.json',variant)
    common=results['hf_reference_r1']
    for result in results.values():
        for key in ('input_manifest_sha256','math_policy','common_framework'):assert result[key]==common[key]
    quality={}
    for role in manifest['roles']:
        name=role['name'];reference=results['hf_reference_r1' if role['order']==1 else 'hf_reference_r2']
        rows=results[name]['ranks'][0];ref=reference['ranks'][0]
        train=[abs(a['global_loss']-b['global_loss']) for a,b in zip(rows['rows'][:16],ref['rows'][:16])]
        validation=[dict(after_updates=a['after_updates'],loss=a['loss'],reference_loss=b['loss'],absolute_drift=abs(a['loss']-b['loss']),passed=abs(a['loss']-b['loss'])<.002) for a,b in zip(rows['full_validation'],ref['full_validation'])]
        quality[name]=dict(training_mean_abs_loss_drift=statistics.mean(train),training_max_abs_loss_drift=max(train),training_loss_drifts=train,full_validation=validation,short_quality_contract_passed=statistics.mean(train)<.002 and max(train)<.01 and all(v['passed'] for v in validation))
        base=timing['hf_reference_r1' if role['order']==1 else 'hf_reference_r2']['median_seconds'];timing[name]['step_reduction_vs_HF']=1-timing[name]['median_seconds']/base
        base=timing['triton_base_r1' if role['order']==1 else 'triton_base_r2']['median_seconds'];timing[name]['step_reduction_vs_Triton_base']=1-timing[name]['median_seconds']/base
    ablations=[]
    for variant in ('ablate_attention','ablate_norm_rope','ablate_mlp','ablate_linear','ablate_loss'):
        ablations.append(dict(variant=variant,orders=[dict(order=order,optimized_seconds=timing[f'triton_optimized_r{order}']['median_seconds'],ablated_seconds=timing[f'{variant}_r{order}']['median_seconds'],ablation_step_increase_fraction=timing[f'{variant}_r{order}']['median_seconds']/timing[f'triton_optimized_r{order}']['median_seconds']-1) for order in (1,2)]))
    strict=all(g['all_parameter_gradient_gate_passed'] and g['forward_gate_passed'] for group in gradient_checks.values() for g in group)
    report=dict(complete=True,audit_passed=True,model='Qwen/Qwen3-8B',world_size=4,layers=36,roles=manifest['roles'],trainability_and_state_checks_passed=True,strict_BF16_numeric_gate_passed=strict,all_short_quality_contracts_passed=all(q['short_quality_contract_passed'] for q in quality.values()),CUDA_training_trace_coverage_passed=all(t.get('coverage_passed',True) for t in traces.values()),quality=quality,gradient_checks=gradient_checks,training_CUDA_traces=traces,timing=timing,ablations=ablations,performance_adopted=False,scope=__doc__,sources={str(p):sha(p) for p in [Path(__file__),D/'manifest.json',*[D/role['name']/'result.json' for role in manifest['roles']]]})
    (D/'independent_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('gradient_checks','training_CUDA_traces','sources')},indent=2),flush=True)
if __name__=='__main__':main()
