"""Independent CPU qualification of the complete two-order peer-RS experiment.

Revalidates original archived references and recorded all-coordinate error
arithmetic. Candidate dense gradients are not archived or independently replayed.
No experiment is adopted until ALL8 roles pass the original gates and the
candidate wins >=3% over all three matched-order valid controls in BOTH orders.
"""
import argparse,json,math,statistics
from pathlib import Path
import torch
from qwen3_triton.gradient_archive import load,sha
from training.provenance import source_snapshot
from qwen3_peer_v13_contract import audit_trace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert out.is_relative_to(Path('/data1/tzh'))
    manifest=json.loads((out/'manifest.json').read_text());root=Path(manifest['source_root'])
    assert manifest['minimum_reduction_each_order']==.03 and manifest['memory_cap_GiB']==41.875
    assert manifest['peer_external_reserve_bytes']==512*2**20 and manifest['peer_torch_allocator_cap_GiB']==41.375
    assert manifest['quality_steps']==16 and manifest['warmup']==3 and manifest['measured']==8 and manifest['updates']==27
    assert source_snapshot(root)==manifest['core_snapshot']
    for path,digest in manifest['sources'].items():assert sha(path)==digest,path
    torch.set_num_threads(2)
    reference=Path(manifest['reference_dir']);ref=json.loads((reference/'result.json').read_text())
    assert sha(reference/'result.json')==manifest['reference_result_sha256'] and ref['variant']=='hf' and ref['layers']==36
    assert ref['input_manifest_sha256']==manifest['input_identity'] and ref['math_policy']['allow_bf16_reduced_precision_reduction'] is False
    canonical=[];norms=[];reference_hashes={}
    for rank in range(4):
        path=reference/f'raw_rank{rank}.pt.gz';digest=sha(path)
        assert digest==ref['ranks'][rank]['reference_archive']['sha256'];reference_hashes[str(path)]=digest
        raw=load(path)['1'];assert raw['ids'].shape==(1,512) and raw['logits'].shape==(1,512,151936)
        assert len(raw['gradients'])==399
        assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in raw['gradients'].values())
        canonical.append(set(raw['gradients']));norms.append({name:float(t.norm()) for name,t in raw['gradients'].items()});del raw
    history=json.loads((out/'attempts.json').read_text());roles=manifest['roles']
    assert [r['name'] for r in roles]==['native_r1','V12_r1','HF_peer_r1','candidate_r1','candidate_r2','HF_peer_r2','V12_r2','native_r2']
    assert [r['number'] for r in history]==list(range(1,len(history)+1))
    assert [r['role'] for r in history if r['accepted']]==[r['name'] for r in roles]
    evidence={};resources=[];data={};timing={};traces={};numerics={};memory={};transport={}
    plan=json.loads((root/'results/qwen3_triton_v10_gemm_plan_001.json').read_text())
    for role in roles:
        name=role['name'];attempts=[r for r in history if r['role']==name];clean=[r for r in attempts if not r['resource_contaminated']]
        assert len(clean)==1 and clean[0] is attempts[-1] and clean[0]['accepted'] and clean[0]['returncode']==0
        assert (out/name).resolve()==(out/clean[0]['folder']).resolve()
        for row in attempts:
            folder=out/row['folder'];assert sha(folder/'monitor.json')==row['monitor_sha256'] and sha(folder/'worker.log')==row['log_sha256']
            evidence[str(folder/'monitor.json')]=row['monitor_sha256'];evidence[str(folder/'worker.log')]=row['log_sha256']
        monitor=json.loads((out/name/'monitor.json').read_text());owned=set(monitor['known_owned_pids'])
        assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors'] and monitor['samples']
        assert all(not s.get('foreign_pids') and 'observation_error' not in s and all(p['pid'] in owned for p in s['apps']) for s in monitor['samples'])
        resources.append(dict(role=name,accepted_attempt=clean[0]['number'],monitor_sha256=sha(out/name/'monitor.json')))
        path=out/name/'run/result.json';d=json.loads(path.read_text());assert sha(path)==clean[0]['result_sha256']
        assert d['complete'] and not d['screen_only'] and d['model']=='Qwen/Qwen3-8B' and d['layers']==36 and d['world_size']==4
        assert d['variant']==role['variant'] and d['input_manifest_sha256']==manifest['input_identity'] and d['input_root']==manifest['input_root']
        assert d['math_policy']==dict(compute='BF16',master_gradient_moments='FP32',TF32=False,allow_bf16_reduced_precision_reduction=False)
        expected=dict(checkpoint_every=1,keep_full_layers=36,skip_embedding_backward_gather=True,retain_lm_head=not role['vocab_ce'],
            vocab_parallel=role['vocab_ce'],custom_collectives=role['peer'],custom_collective_schedule=role['vocab_ce'],head_gradient_preaveraged=role['vocab_ce'],
            fused_vocab_dw=role['vocab_ce'],vocab_dw_config=[64,128,32,8,8,3] if role['vocab_ce'] else None,
            optimizer='native_fused_AdamW',optimizer_groups='decoder+head_in_all_arms',batch_per_rank=1,sequence=512)
        if role['peer']:expected.update(gradient_transport='checked lossless BF16 staging; FP32 reduction and output',external_memory_budget_bytes=512*2**20,communication_experiment='peer_RS_v13')
        assert d['common_framework']==expected
        for path,digest in d['sources'].items():assert sha(path)==digest,path
        rank_numerics=[];rank_memory=[];rank_transport=[];samples=None
        assert len(d['ranks'])==4
        for rank,row in enumerate(d['ranks']):
            assert row['rank']==rank and row['complete'] and row['updates']==27 and row['parameter_count']==row['optimizer_state_count']==399
            assert row['all_master_grad_moment_finite_fp32']
            assert [r['step'] for r in row['rows']]==list(range(1,28))
            assert all(r['measured']==(r['step']>19) for r in row['rows'])
            assert all(math.isfinite(r['global_loss']) and math.isfinite(r['wall_seconds']) and r['wall_seconds']>0 for r in row['rows'])
            own_samples=[r['wall_seconds'] for r in row['rows'] if r['measured']];assert len(own_samples)==8
            if samples is None:samples=own_samples
            assert own_samples==samples and [r['global_loss'] for r in row['rows']]==[r['global_loss'] for r in d['ranks'][0]['rows']]
            assert [v['after_updates'] for v in row['full_validation']]==[0,16] and row['full_validation']==d['ranks'][0]['full_validation']
            assert all(v['blocks']==512 and v['tokens']==261632 and v['batch_per_rank']==4 for v in row['full_validation'])
            assert row['vocab_head_calls']==(dict(forward=92,backward=28) if role['vocab_ce'] else None)
            alloc=row['overwrite_outputs'];assert not alloc['enabled'] and alloc['calls']==0 and alloc['deterministic_algorithms_enabled'] and alloc['deterministic_fill_enabled']
            assert len(row['gradient_comparisons'])==1;g=row['gradient_comparisons'][0];strict=role['variant']=='hf'
            assert g['complete'] and g['strict_native_control']==strict and g['reference_result_sha256']==manifest['reference_result_sha256']
            assert len(g['rows'])==399 and {r['name'] for r in g['rows']}==canonical[rank]
            failures=[]
            for r in g['rows']:
                assert all(math.isfinite(r[k]) and r[k]>=0 for k in ['absolute_error_norm','reference_norm','relative_l2','threshold'])
                assert r['reference_norm']==norms[rank][r['name']]
                threshold=(.0001 if strict else .06)*max(r['reference_norm'],1e-8)+(1e-7 if strict else 1e-6)
                assert r['threshold']==threshold and r['passed']==(r['absolute_error_norm']<threshold)
                assert r['relative_l2']==r['absolute_error_norm']/max(r['reference_norm'],1e-8)
                if not r['passed']:failures.append(r)
            error=sum(r['absolute_error_norm']**2 for r in g['rows']);norm=sum(r['reference_norm']**2 for r in g['rows'])
            assert g['global_relative_l2']==(error/max(norm,1e-16))**.5
            forward=g['logits_relative_l2']<.025 and g['loss_abs_error']<.01 and (g['forward_bitwise_equal'] or not strict)
            numerical=not failures and forward
            assert g['passed']==(not failures) and g['forward_gate_passed']==forward and g['numerical_gate_passed']==numerical
            rank_numerics.append(dict(rank=rank,passed=numerical,all399=True,gradient_relative_l2=g['global_relative_l2'],forward_bitwise_equal=g['forward_bitwise_equal'],failed_gradients=failures))
            extra=(512*2**20 if role['peer'] else 0)/2**30
            assert row['effective_training_peak']=={k:v+extra for k,v in row['measured_training_peak'].items()}
            rank_memory.append(max(row['effective_training_peak'].values())<=41.875)
            comm=row['communication']
            if role['peer']:
                assert comm['wire_dtype']=='torch.bfloat16' and comm['accumulate_output_dtype']=='torch.float32'
                assert comm['mode']=='peer_bf16_checked' and comm['peer_calls']==comm['calls']==36*28
                assert comm['native_calls']==(3 if role['variant']=='hf' else 2)*28
                assert comm['checked_elements']==comm['peer_calls']*192946432
                assert comm['external_memory_budget_bytes']==512*2**20 and comm['external_storage_accounted_in_effective_budget']
                assert comm['observed_extra_process_bytes']<=512*2**20 and comm['handle_buffer_size']+comm['signal_pad_size']<=512*2**20
                assert comm['torch_allocator_cap_GiB']==41.375 and comm['total_effective_cap_GiB']==41.875
                rank_transport.append(comm['lossless'] and comm['lossless_violation_flag']==0 and comm['guard_positive_negative_selftest_passed'])
            else:assert comm==dict(mode='native');rank_transport.append(True)
        data[name]=d;numerics[name]=rank_numerics;memory[name]=rank_memory;transport[name]=rank_transport
        timing[name]=dict(median_seconds=statistics.median(samples),samples_seconds=samples,
            torch_allocated_GiB=max(r['measured_training_peak']['allocated_gib'] for r in d['ranks']),
            torch_reserved_GiB=max(r['measured_training_peak']['reserved_gib'] for r in d['ranks']),
            effective_allocated_GiB=max(r['effective_training_peak']['allocated_gib'] for r in d['ranks']),
            effective_reserved_GiB=max(r['effective_training_peak']['reserved_gib'] for r in d['ranks']))
        traces[name]=audit_trace(out/name/'run/training_trace.json',role['variant'],role['vocab_ce'],plan if role['variant']=='triton_optimized' else None,peer=role['peer'])
        assert traces[name]==clean[0]['actual_CUDA'] and traces[name]['trace_sha256']==clean[0]['trace_sha256']
        for pth in [out/name/'run/result.json',out/name/'run/training_trace.json']:evidence[str(pth)]=sha(pth)
    quality={};recorded_equal={}
    for role in roles:
        name=role['name'];row=data[name]['ranks'][0];base=data[f'native_r{role["order"]}']['ranks'][0]
        drift=[abs(x['global_loss']-y['global_loss']) for x,y in zip(row['rows'][:16],base['rows'][:16])]
        val=[abs(x['loss']-y['loss']) for x,y in zip(row['full_validation'],base['full_validation'])]
        quality[name]=dict(mean_loss_drift=statistics.mean(drift),max_loss_drift=max(drift),validation_drifts=val,passed=statistics.mean(drift)<.002 and max(drift)<.01 and max(val)<.002)
    comparisons=[]
    for order in [1,2]:
        seconds={arm:timing[f'{arm}_r{order}']['median_seconds'] for arm in ['native','V12','HF_peer','candidate']}
        gains={arm:1-seconds['candidate']/seconds[arm] for arm in ['native','V12','HF_peer']}
        comparisons.append(dict(order=order,seconds=seconds,candidate_reductions=gains,shared_peer_reduction_on_HF=1-seconds['HF_peer']/seconds['native']))
        current=data[f'candidate_r{order}']['ranks'][0];base=data[f'V12_r{order}']['ranks'][0]
        recorded_equal[str(order)]=dict(all27_losses=[r['global_loss'] for r in current['rows']]==[r['global_loss'] for r in base['rows']],
            validation=current['full_validation']==base['full_validation'],
            recorded_first_gradient_metrics=[r['gradient_comparisons'] for r in data[f'candidate_r{order}']['ranks']]==[r['gradient_comparisons'] for r in data[f'V12_r{order}']['ranks']])
    gates=dict(numerics=all(c['passed'] for rows in numerics.values() for c in rows),short_quality=all(v['passed'] for v in quality.values()),
        effective_memory=all(all(v) for v in memory.values()),lossless_transport=all(all(v) for v in transport.values()),actual_CUDA=all(v['passed'] for v in traces.values()),
        first_clean_resource_selection=True,original_sources_and_reference_archives=True)
    performance=all(g>=.03 for c in comparisons for g in c['candidate_reductions'].values())
    report=dict(complete=True,audit_passed=True,qualified_training_improvement=all(gates.values()) and performance,
        gates=gates,performance_rule_passed=performance,comparisons=comparisons,timing=timing,numerics=numerics,
        quality=quality,memory=memory,lossless_transport=transport,traces=traces,resource_selection=resources,
        recorded_trajectory_equal_to_V12=recorded_equal,all_original_thresholds_unchanged=True,default_changed=False,
        reference_archive_sha256=reference_hashes,raw_evidence_sha256=evidence,
        sources={str(p):sha(p) for p in [Path(__file__),out/'manifest.json',out/'attempts.json',root/'scripts/qwen3_peer_v13_contract.py',root/'scripts/qwen3_vocab_v12_contract.py',root/'scripts/qwen3_training_v11_contract.py']},
        frozen_sources=manifest['sources'],scope=__doc__,limitations='16 quality updates, not long convergence. Rank0 post27 forward/backward trace excludes optimizer; full-step timings include optimizer. Candidate dense gradients not archived, reference archives and recorded arithmetic rechecked. Two orders are not statistical significance. Effective memory includes a conservative512MiB external reserve, not NVML total. No inference claim.')
    with (out/'independent_review.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(qualified=report['qualified_training_improvement'],gates=gates,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
