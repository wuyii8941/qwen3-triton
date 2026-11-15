"""Independent CPU review of full Qwen3-8B vocabulary CE comparisons.

Candidate dense gradients are compared at runtime and not archived. This
review replays original reference archives and verifies recorded arithmetic,
source identity, complete quality runs, real CUDA traces and timing samples.
"""
from pathlib import Path
import argparse,json,math,statistics
import torch
from qwen3_triton.gradient_archive import load,sha
from training.provenance import source_snapshot
from qwen3_vocab_v12_contract import audit_trace


def monitor_clean(path):
    a=json.loads(path.read_text());owned=set(a['known_owned_pids'])
    assert a['returncode']==0 and not a['foreign_process_seen'] and not a['observation_errors'] and a['samples']
    assert all(not s.get('foreign_pids') and 'observation_error' not in s and
               all(p['pid'] in owned for p in s['apps']) for s in a['samples'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();assert out.is_relative_to(Path('/data1/tzh'))
    torch.set_num_threads(2);manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['minimum_reduction_each_order']==.03 and manifest['memory_cap_GiB']==41.875
    assert manifest['checkpoint_every'] in (0,1,2)
    assert source_snapshot(Path(manifest['source_root']))==manifest['core_snapshot']
    for path,digest in manifest['sources'].items():assert sha(path)==digest,path
    reference=Path(manifest['reference_dir']);ref=json.loads((reference/'result.json').read_text())
    assert ref['variant']=='hf' and ref['layers']==36 and sha(reference/'result.json')==manifest['reference_result_sha256']
    canonical=[]
    for rank in range(4):
        path=reference/f'raw_rank{rank}.pt.gz'
        assert sha(path)==ref['ranks'][rank]['reference_archive']['sha256']
        raw=load(path)['1'];assert raw['ids'].shape==(1,512) and raw['logits'].shape==(1,512,151936)
        assert len(raw['gradients'])==399
        assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in raw['gradients'].values())
        canonical.append(set(raw['gradients']));del raw
    results={};numerics={};timing={};traces={};evidence={}
    roles=manifest['roles'];arms=['native','triton_control','HF_custom_head','candidate']
    if manifest['checkpoint_every']!=1:arms.insert(2,'head_cp1')
    assert [r['name'] for r in roles]==[a+'_r1' for a in arms]+[a+'_r2' for a in reversed(arms)]
    plan_path=Path(manifest['gemm_plan']['path']);assert sha(plan_path)==manifest['gemm_plan']['sha256']
    plan=json.loads(plan_path.read_text())
    for role in roles:
        name=role['name'];monitor_clean(out/(name+'.gpu.json'))
        path=out/name/'result.json';a=json.loads(path.read_text());evidence[str(path)]=sha(path)
        assert a['complete'] and not a['screen_only'] and a['experiment']=='vocabulary_CE_v12'
        assert a['variant']==role['variant'] and a['model']=='Qwen/Qwen3-8B' and a['layers']==36 and a['world_size']==4
        assert a['input_root']==manifest['input_root'] and a['input_manifest_sha256']==manifest['input_identity']
        assert a['math_policy']==dict(compute='BF16',master_gradient_moments='FP32',TF32=False,allow_bf16_reduced_precision_reduction=False)
        expected=dict(checkpoint_every=role['checkpoint_every'],keep_full_layers=36,skip_embedding_backward_gather=True,
            retain_lm_head=not role['vocab_ce'],vocab_parallel=role['vocab_ce'],custom_collectives=False,
            custom_collective_schedule=role['vocab_ce'],head_gradient_preaveraged=role['vocab_ce'],
            optimizer='native_fused_AdamW',optimizer_groups='decoder+head_in_all_arms',batch_per_rank=1,sequence=512)
        assert a['common_framework']==expected
        for path,digest in a['sources'].items():assert sha(path)==digest,path
        tuned=role['variant']=='triton_optimized'
        assert (a['gemm_plan']['sha256']==manifest['gemm_plan']['sha256']) if tuned else a['gemm_plan'] is None
        checks=[];samples=None
        for rank,row in enumerate(a['ranks']):
            assert row['rank']==rank and row['complete'] and row['updates']==27
            assert row['parameter_count']==row['optimizer_state_count']==399 and row['all_master_grad_moment_finite_fp32']
            assert row['vocab_head_calls']==(dict(forward=92,backward=28) if role['vocab_ce'] else None)
            assert [r['step'] for r in row['rows']]==list(range(1,28))
            assert [v['after_updates'] for v in row['full_validation']]==[0,16]
            assert all(v['blocks']==512 and v['tokens']==261632 and v['batch_per_rank']==4 for v in row['full_validation'])
            assert max(row['measured_training_peak'].values())<=41.875
            assert all(math.isfinite(r['global_loss']) and math.isfinite(r['wall_seconds']) and r['wall_seconds']>0 for r in row['rows'])
            current=[r['wall_seconds'] for r in row['rows'] if r['measured']];assert len(current)==8
            if samples is None:samples=current
            else:assert current==samples
            assert [r['global_loss'] for r in row['rows']]==[r['global_loss'] for r in a['ranks'][0]['rows']]
            assert row['full_validation']==a['ranks'][0]['full_validation']
            assert len(row['gradient_comparisons'])==1;g=row['gradient_comparisons'][0]
            strict=role['arm']=='native'
            assert g['complete'] and g['strict_native_control']==strict and g['reference_result_sha256']==manifest['reference_result_sha256']
            assert len(g['rows'])==399 and {r['name'] for r in g['rows']}==canonical[rank]
            for r in g['rows']:
                assert all(math.isfinite(r[k]) and r[k]>=0 for k in ['absolute_error_norm','reference_norm','relative_l2','threshold'])
                threshold=(.0001 if strict else .06)*max(r['reference_norm'],1e-8)+(1e-7 if strict else 1e-6)
                assert r['threshold']==threshold and r['passed']==(r['absolute_error_norm']<threshold)
                assert r['relative_l2']==r['absolute_error_norm']/max(r['reference_norm'],1e-8)
            grad=all(r['passed'] for r in g['rows'])
            forward=g['logits_relative_l2']<.025 and g['loss_abs_error']<.01 and (g['forward_bitwise_equal'] or not strict)
            error=sum(r['absolute_error_norm']**2 for r in g['rows']);norm=sum(r['reference_norm']**2 for r in g['rows'])
            assert g['global_relative_l2']==(error/max(norm,1e-16))**.5
            assert g['passed']==grad and g['forward_gate_passed']==forward and g['numerical_gate_passed']==(grad and forward)
            checks.append(dict(rank=rank,passed=grad and forward,gradient_relative_l2=g['global_relative_l2'],
                logits_relative_l2=g['logits_relative_l2'],loss_abs_error=g['loss_abs_error'],forward_bitwise_equal=g['forward_bitwise_equal']))
        results[name]=a;numerics[name]=checks
        timing[name]=dict(median_seconds=statistics.median(samples),samples_seconds=samples,
            allocated_GiB=max(r['measured_training_peak']['allocated_gib'] for r in a['ranks']),
            reserved_GiB=max(r['measured_training_peak']['reserved_gib'] for r in a['ranks']))
        traces[name]=audit_trace(out/name/'training_trace.json',role['variant'],role['vocab_ce'],plan if tuned else None,role['checkpoint_every'])
    quality={}
    for role in roles:
        name=role['name'];row=results[name]['ranks'][0];control=results[f"native_r{role['order']}"]['ranks'][0]
        drift=[abs(a['global_loss']-b['global_loss']) for a,b in zip(row['rows'][:16],control['rows'][:16])]
        validation=[abs(a['loss']-b['loss']) for a,b in zip(row['full_validation'],control['full_validation'])]
        quality[name]=dict(mean_loss_drift=statistics.mean(drift),max_loss_drift=max(drift),validation_drifts=validation,
            passed=statistics.mean(drift)<.002 and max(drift)<.01 and max(validation)<.002)
    comparisons=[]
    for order in [1,2]:
        t={arm:timing[f'{arm}_r{order}']['median_seconds'] for arm in arms}
        comparisons.append(dict(order=order,seconds=t,reduction_vs_native_HF=1-t['candidate']/t['native'],
            reduction_vs_triton_V11=1-t['candidate']/t['triton_control'],
            reduction_vs_HF_custom_head=1-t['candidate']/t['HF_custom_head'],
            checkpoint_incremental_reduction=1-t['candidate']/t['head_cp1'] if 'head_cp1' in t else None))
    gates=dict(numerics=all(c['passed'] for rows in numerics.values() for c in rows),
        short_quality=all(q['passed'] for q in quality.values()),
        actual_CUDA_and_collectives=all(t['passed'] for t in traces.values()))
    performance=all(c['reduction_vs_native_HF']>=.03 and c['reduction_vs_triton_V11']>=.03 for c in comparisons)
    report=dict(complete=True,audit_passed=True,gates=gates,performance_rule_passed=performance,
        qualified_training_improvement=all(gates.values()) and performance,default_changed=False,
        checkpoint_every=manifest['checkpoint_every'],comparisons=comparisons,timing=timing,numerics=numerics,
        quality=quality,traces=traces,sources={str(Path(__file__).resolve()):sha(Path(__file__)),str(out/'manifest.json'):sha(out/'manifest.json'),**evidence},
        limitations='Two orders are descriptive, not statistical significance. Dense candidate gradients were compared at runtime, not archived. Independent CPU replay verifies reference archives and recorded error arithmetic. Sixteen quality updates do not prove convergence. HF custom head requires the same bespoke Triton loss, parameter ownership and collective schedule. No inference claim.')
    with (out/'independent_review.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(gates=gates,qualified_training_improvement=report['qualified_training_improvement'],comparisons=comparisons),indent=2),flush=True)


if __name__=='__main__':main()
