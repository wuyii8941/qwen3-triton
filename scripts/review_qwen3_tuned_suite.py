"""Independent CPU review of the paired V9/tuned/HF training comparison.

Replay the native archive, recompute recorded numerical thresholds, inspect
actual CUDA kernels and launch geometries, and compare complete optimizer
steps and short-training quality. Dense candidate gradients are not archived.
"""
from pathlib import Path
from collections import Counter
import argparse,json,math,statistics
import torch
from qwen3_triton.gradient_archive import sha,load
from review_qwen3_triton_suite import trace_report


def launch_geometry(path,plan):
    data=json.loads(path.read_text())
    events=[e for e in data['traceEvents'] if e.get('cat')=='kernel']
    table={('gemm',e['M'],e['N'],e['K'],*(v for t in e['strides'] for v in t)):e
           for e in plan['entries'] if e['kind']=='gemm'} if plan else {}
    expected=Counter()
    # Decoder forward is executed twice due to full per-layer checkpointing.
    for output,input_,instances,fused,decoder in ((4096,4096,72,False,True),
            (1024,4096,72,False,True),(12288,4096,72,True,True),
            (4096,12288,36,False,True),(151936,4096,1,False,False)):
        cases=[]
        if not fused:cases.append((512,output,input_,input_,1,1,input_,instances*(2 if decoder else 1)))
        cases.extend(((512,input_,output,output,1,input_,1,instances),
                      (output,input_,512,1,output,input_,1,instances)))
        for m,n,k,am,ak,bk,bn,count in cases:
            entry=table.get(('gemm',m,n,k,am,ak,bk,bn))
            if entry:assert entry['family']=='tiled' and entry['splits']==1,'Extend the trace geometry contract before accepting split-K'
            tm,tn,tk,group,warps,stages=entry['config'] if entry else (64,64,32,8,4,3)
            expected[(((m+tm-1)//tm)*((n+tn-1)//tn),warps*32)]+=count
    assert sum(expected.values())==867
    observed=Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name']=='_gemm')
    expected_geometry=Counter({((grid,1,1),(threads,1,1)):count for (grid,threads),count in expected.items()})
    gate=next((e for e in plan['entries'] if e['kind']=='gate_up' and (e['M'],e['N'],e['K'])==(512,12288,4096)),None) if plan else None
    tm,tn,tk,group,warps,stages=gate['config'] if gate else (64,64,32,8,4,3)
    gate_expected=Counter({(((512+tm-1)//tm*((12288+tn-1)//tn),1,1),(warps*32,1,1)):72})
    gate_observed=Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name']=='_gate_up')
    return dict(passed=expected_geometry==observed and gate_expected==gate_observed,
        expected_gemm={str(k):v for k,v in expected_geometry.items()},
        observed_gemm={str(k):v for k,v in observed.items()},
        expected_gate_up={str(k):v for k,v in gate_expected.items()},
        observed_gate_up={str(k):v for k,v in gate_observed.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    assert not root.is_relative_to(Path('/home/tzh'))
    torch.set_num_threads(2)
    manifest=json.loads((root/'manifest.json').read_text())
    for path,h in manifest['sources'].items():assert sha(path)==h
    forward=('hf','v9','v10_all','v10') if 'v10_all' in manifest['plans'] else ('hf','v9','v10')
    assert len(manifest['roles'])==2*len(forward)
    assert [(r['arm'],r['order']) for r in manifest['roles']]==[(a,o) for o,arms in ((1,forward),(2,tuple(reversed(forward)))) for a in arms]
    reference=Path(manifest['reference_dir']);reference_result=json.loads((reference/'result.json').read_text())
    assert sha(reference/'result.json')==manifest['reference_result_sha256']
    canonical_names=[]
    for rank in range(4):
        path=reference/f'raw_rank{rank}.pt.gz'
        assert sha(path)==reference_result['ranks'][rank]['reference_archive']['sha256']
        raw=load(path)['1']
        assert raw['ids'].shape==(1,512) and raw['logits'].shape==(1,512,151936)
        assert len(raw['gradients'])==399
        assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in raw['gradients'].values())
        canonical_names.append(set(raw['gradients']));del raw
    plans={}
    for name,record in manifest['plans'].items():
        assert sha(record['path'])==record['sha256']
        plans[name]=json.loads(Path(record['path']).read_text())
    results={};timings={};numerics={};traces={}
    for role in manifest['roles']:
        name=role['name'];arm=role['arm']
        monitor=json.loads((root/(name+'.gpu.json')).read_text())
        assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
        result=json.loads((root/name/'result.json').read_text())
        assert result['complete'] and result['model']=='Qwen/Qwen3-8B' and result['layers']==36 and result['world_size']==4
        assert result['variant']==role['variant']
        assert result['performance_adopted'] is False and result['numerical_equivalence_qualified'] is False
        assert result['math_policy']==dict(compute='BF16',master_gradient_moments='FP32',TF32=False,allow_bf16_reduced_precision_reduction=False)
        for path,h in result['sources'].items():assert sha(path)==h
        if role['tuned']:assert result['gemm_plan']['sha256']==manifest['plans'][arm]['sha256']
        else:assert result['gemm_plan'] is None
        samples=None;checks=[]
        for rank,row in enumerate(result['ranks']):
            assert row['gemm_plan']==result['gemm_plan']
            assert row['rank']==rank and row['complete'] and row['parameter_count']==row['optimizer_state_count']==399
            assert row['updates']==27 and row['all_master_grad_moment_finite_fp32']
            assert [r['step'] for r in row['rows']]==list(range(1,28))
            assert [v['after_updates'] for v in row['full_validation']]==[0,16]
            assert all(v['blocks']==512 and v['tokens']==261632 and v['batch_per_rank']==4 for v in row['full_validation'])
            assert max(row['measured_training_peak'].values())<=41.875
            assert all(math.isfinite(r['global_loss']) and math.isfinite(r['wall_seconds']) and r['wall_seconds']>0 for r in row['rows'])
            current=[r['wall_seconds'] for r in row['rows'] if r['measured']]
            assert len(current)==8
            if samples is None:samples=current
            else:assert samples==current
            assert [r['global_loss'] for r in row['rows']]==[r['global_loss'] for r in result['ranks'][0]['rows']]
            assert len(row['gradient_comparisons'])==1
            comparison=row['gradient_comparisons'][0]
            assert comparison['complete'] and len(comparison['rows'])==399
            assert {g['name'] for g in comparison['rows']}==canonical_names[rank]
            assert comparison['reference_result_sha256']==manifest['reference_result_sha256']
            strict=arm=='hf';assert comparison['strict_native_control']==strict
            for gradient in comparison['rows']:
                assert all(math.isfinite(gradient[k]) and gradient[k]>=0 for k in ('absolute_error_norm','reference_norm','relative_l2','threshold'))
                threshold=(.0001 if strict else .06)*max(gradient['reference_norm'],1.e-8)+(1.e-7 if strict else 1.e-6)
                assert threshold==gradient['threshold']
                assert gradient['passed']==(gradient['absolute_error_norm']<threshold)
            passed=all(g['passed'] for g in comparison['rows'])
            forward=comparison['logits_relative_l2']<.025 and comparison['loss_abs_error']<.01 and (comparison['forward_bitwise_equal'] or not strict)
            assert comparison['passed']==passed and comparison['forward_gate_passed']==forward
            error2=sum(g['absolute_error_norm']**2 for g in comparison['rows'])
            norm2=sum(g['reference_norm']**2 for g in comparison['rows'])
            assert comparison['global_relative_l2']==(error2/max(norm2,1.e-16))**.5
            checks.append(dict(rank=rank,parameter_count=399,passed=passed and forward,
                failed_parameters=[g for g in comparison['rows'] if not g['passed']],
                global_gradient_relative_l2=comparison['global_relative_l2'],
                logits_relative_l2=comparison['logits_relative_l2'],loss_abs_error=comparison['loss_abs_error'],
                forward_bitwise_equal=comparison['forward_bitwise_equal']))
        numerics[name]=checks;results[name]=result
        timings[name]=dict(arm=arm,order=role['order'],samples_seconds=samples,
            median_seconds=statistics.median(samples),peak_allocated_gib=max(r['measured_training_peak']['allocated_gib'] for r in result['ranks']),
            peak_reserved_gib=max(r['measured_training_peak']['reserved_gib'] for r in result['ranks']))
        trace=trace_report(root/name/'training_trace.json',role['variant'])
        if arm!='hf':trace['launch_geometry']=launch_geometry(root/name/'training_trace.json',plans[arm] if role['tuned'] else None)
        traces[name]=trace
    common=results['hf_r1']
    for result in results.values():
        for key in ('input_manifest_sha256','math_policy','common_framework'):assert result[key]==common[key]
    quality={}
    for role in manifest['roles']:
        name=role['name'];row=results[name]['ranks'][0]
        ref=results[f'hf_r{role["order"]}']['ranks'][0]
        train=[abs(a['global_loss']-b['global_loss']) for a,b in zip(row['rows'][:16],ref['rows'][:16])]
        validation=[dict(after_updates=a['after_updates'],loss=a['loss'],reference_loss=b['loss'],absolute_drift=abs(a['loss']-b['loss'])) for a,b in zip(row['full_validation'],ref['full_validation'])]
        passed=statistics.mean(train)<.002 and max(train)<.01 and all(v['absolute_drift']<.002 for v in validation)
        quality[name]=dict(passed=passed,training_mean_drift=statistics.mean(train),training_max_drift=max(train),training_drifts=train,validation=validation)
    comparisons=[]
    for order in (1,2):
        hf,v9=(timings[f'{arm}_r{order}']['median_seconds'] for arm in ('hf','v9'))
        for arm in manifest['plans']:
            candidate=timings[f'{arm}_r{order}']['median_seconds']
            comparisons.append(dict(arm=arm,order=order,hf_seconds=hf,v9_seconds=v9,candidate_seconds=candidate,
                latency_reduction_vs_HF=1-candidate/hf,latency_reduction_vs_V9=1-candidate/v9))
    gates=dict(numerical=all(c['passed'] for rows in numerics.values() for c in rows),
        short_quality=all(row['passed'] for row in quality.values()),
        CUDA_coverage=all(t.get('coverage_passed',True) and t.get('launch_geometry',{}).get('passed',True) for t in traces.values()))
    report=dict(complete=True,audit_passed=True,all_acceptance_gates_passed=all(gates.values()),gates=gates,
        roles=manifest['roles'],timing=timings,comparisons=comparisons,numerics=numerics,
        quality=quality,traces=traces,performance_adopted=False,
        limitation='Two orders are descriptive; no statistical significance or convergence/inference claim. All candidate gradient coordinates were compared at runtime; dense candidate vectors were not saved.',
        sources={str(p):sha(p) for p in [Path(__file__),root/'manifest.json',*[root/r['name']/'result.json' for r in manifest['roles']]]},scope=__doc__)
    (root/'independent_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,gates=gates,comparisons=comparisons),indent=2),flush=True)


if __name__=='__main__':main()
