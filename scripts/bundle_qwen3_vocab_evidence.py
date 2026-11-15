"""Collect completed V12 evidence and derive explicit full-model flags.

Never publishes weights, corpus, caches, dense gradients, logs or raw traces.
The recommendation policy is declared before formal candidate timing, and
requires the same original numerical/quality/3%-in-both-orders gates.
"""
from pathlib import Path
import argparse,json,statistics
from qwen3_triton.gradient_archive import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--cohort',default='formal_003')
    p.add_argument('--policy',default='fusion_recommendation_policy_002.json')
    p.add_argument('--results',type=Path,required=True);args=p.parse_args()
    base=args.artifacts.resolve();destination=args.results.resolve()
    assert Path(args.cohort).name==args.cohort
    formal=base/args.cohort
    assert destination.is_relative_to(Path('/data1/tzh')) and destination.is_dir()
    review=json.loads((formal/'independent_review.json').read_text())
    assert review['complete'] and review['audit_passed']
    methods=json.loads((formal/'per_method_review.json').read_text())
    assert methods['complete'] and methods['audit_passed'] and methods['all_original_thresholds_unchanged']
    assert methods['recommended_method_qualified'] and methods['native_HF_and_V11_controls_qualified']
    assert methods['original_cohort_review_sha256']==sha(formal/'independent_review.json')
    manifest=json.loads((formal/'manifest.json').read_text())
    assert manifest['memory_cap_GiB']==41.875 and manifest['minimum_reduction_each_order']==.03
    assert manifest['checkpoint_every']==1 and manifest['fused_dw']
    for record in [manifest,review,methods]:
        for path,digest in record['sources'].items():assert sha(path)==digest,path
    assert Path(args.policy).name==args.policy
    policy_path=base/args.policy;policy=json.loads(policy_path.read_text())
    assert policy['complete'] and policy['no_formal_candidate_timing_observed']
    assert policy['formal_manifest_sha256']==sha(formal/'manifest.json')
    resources=json.loads((formal/'resource_review.json').read_text())
    assert resources['complete'] and resources['first_clean_attempt_policy_passed'] and resources['fixed_role_order_passed']
    assert resources['no_timing_based_attempt_selection'] and len(resources['accepted'])==len(manifest['roles'])
    assert resources['manifest_sha256']==sha(formal/'manifest.json') and resources['attempt_history_sha256']==sha(formal/'attempts.json')
    evidence={};results={}
    for role in manifest['roles']:
        name=role['name'];path=formal/name/'result.json';a=json.loads(path.read_text())
        monitor_path=formal/(name+'.gpu.json');monitor=json.loads(monitor_path.read_text())
        assert a['complete'] and not a['screen_only'] and a['model']=='Qwen/Qwen3-8B' and a['layers']==36
        assert a['input_manifest_sha256']==manifest['input_identity'] and len(a['ranks'])==4
        assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
        owned=set(monitor['known_owned_pids']);assert monitor['samples']
        assert all(not s.get('foreign_pids') and 'observation_error' not in s and
                   all(p['pid'] in owned for p in s['apps']) for s in monitor['samples'])
        samples=[r['wall_seconds'] for r in a['ranks'][0]['rows'] if r['measured']]
        assert len(samples)==8 and statistics.median(samples)==review['timing'][name]['median_seconds']
        for rank in a['ranks']:
            assert rank['updates']==27 and rank['parameter_count']==rank['optimizer_state_count']==399
            assert rank['all_master_grad_moment_finite_fp32'] and max(rank['measured_training_peak'].values())<=41.875
            assert samples==[r['wall_seconds'] for r in rank['rows'] if r['measured']]
        trace=formal/name/'training_trace.json';assert sha(trace)==review['traces'][name]['trace_sha256']
        for pth in [path,monitor_path,trace]:evidence[str(pth)]=sha(pth)
        results[name]=a
    for c in review['comparisons']:
        t=c['seconds'];assert c['reduction_vs_native_HF']==1-t['candidate']/t['native']
        assert c['reduction_vs_triton_V11']==1-t['candidate']/t['triton_control']
        assert c['fusion_incremental_reduction']==1-t['candidate']/t['head_unfused']
    unfused_eligible=all(1-c['seconds']['head_unfused']/c['seconds']['native']>=.03 and
                          1-c['seconds']['head_unfused']/c['seconds']['triton_control']>=.03 for c in review['comparisons'])
    fused_consistently_faster=all(c['seconds']['candidate']<=c['seconds']['head_unfused'] for c in review['comparisons'])
    chosen=methods['recommended_arm']
    assert chosen in ['candidate','head_unfused'] and methods['methods'][chosen]['qualified']
    if chosen=='candidate':assert fused_consistently_faster or not unfused_eligible
    comparisons=[]
    for c in review['comparisons']:
        t=c['seconds'];seconds=t[chosen]
        comparisons.append(dict(order=c['order'],seconds=seconds,native_HF_seconds=t['native'],triton_V11_seconds=t['triton_control'],
            reduction_vs_native_HF=1-seconds/t['native'],reduction_vs_triton_V11=1-seconds/t['triton_control']))
    assert all(c['reduction_vs_native_HF']>=.03 and c['reduction_vs_triton_V11']>=.03 for c in comparisons)
    if chosen=='candidate':assert review['fusion_recorded_trajectory_equal']
    for row in json.loads((formal/'attempts.json').read_text()):
        folder=formal/row['folder']
        assert sha(folder/'monitor.json')==row['monitor_sha256'] and sha(folder/'worker.log')==row['log_sha256']
        evidence[str(folder/'monitor.json')]=row['monitor_sha256'];evidence[str(folder/'worker.log')]=row['log_sha256']
    mapping={
        args.cohort+'/independent_review.json':'cohort_review',
        args.cohort+'/per_method_review.json':'method_review',
        args.cohort+'/manifest.json':'cohort_manifest',
        args.cohort+'/resource_review.json':'resource_review',
        args.cohort+'/attempts.json':'attempt_history',
        'full_screen_003/result.json':'full_screen',
        'full_screen_003/manifest.json':'full_screen_manifest',
        'module_002/module/result.json':'fused_head_module',
        'module_002/manifest.json':'fused_head_module_manifest',
        'module_001/module/result.json':'unfused_head_module',
        'module_001/manifest.json':'unfused_head_module_manifest',
        'dw_fusion_screen_001/result.json':'DW_fusion_screen',
        'dw_fusion_screen_001/manifest.json':'DW_fusion_manifest',
        'checkpoint_screen_001/result.json':'checkpoint_screen',
        'checkpoint_screen_001/manifest.json':'checkpoint_manifest',
        'pre_fusion_source_archive_001.json':'checkpoint_source_archive',
        'integration_fix_001.json':'integration_disclosure',
        'pre_admission_audit_edit_disclosure_001.json':'admission_disclosure',
        'admission_controller_failure_001.json':'controller_failure_disclosure',
        'resource_exclusion_001.json':'resource_exclusion',
        'HF_custom_rejection_001.json':'HF_custom_rejection',
        'per_method_adoption_policy_001.json':'method_adoption_policy',
        'CPU_preflight_003.json':'CPU_checks',
        'CPU_resource_policy_checks_001.json':'resource_policy_checks',
        args.policy:'recommendation_policy',
        'environment_001.json':'environment',
        'communication_model_001.json':'communication_model',
    }
    paths={source:destination/f'qwen3_triton_v12_{name}_v1.json' for source,name in mapping.items()}
    strategy_path=destination/'qwen3_triton_v12_strategy_v1.json'
    index_path=destination/'qwen3_triton_v12_evidence_index_v1.json'
    assert all(not p.exists() for p in [*paths.values(),strategy_path,index_path])
    for source in paths:json.loads((base/source).read_text())
    archive=json.loads((base/'pre_fusion_source_archive_001.json').read_text())
    for row in archive['sources']:assert sha(row['archived_path'])==row['sha256']
    for folder in ['full_screen_003','module_002','dw_fusion_screen_001','checkpoint_screen_001']:
        for path in (base/folder).glob('*.gpu.json'):
            monitor=json.loads(path.read_text())
            assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
            evidence[str(path)]=sha(path)
        path=base/folder/'gpu.json'
        if path.exists():
            monitor=json.loads(path.read_text());assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
            evidence[str(path)]=sha(path)
    copies=[]
    for source,path in paths.items():
        raw=(base/source).read_bytes()
        with path.open('xb') as f:f.write(raw)
        assert sha(path)==sha(base/source)
        copies.append(dict(path=path.name,original_path=str(base/source),sha256=sha(path),bytes=len(raw)))
    flags=['--vocab-ce','--keep-full-layers','36','--checkpoint-every','1','--skip-embedding-backward-gather']
    if chosen=='candidate':flags.append('--fused-vocab-dw')
    strategy=dict(complete=True,qualified_training_improvement=True,default_changed=False,recommended_arm=chosen,
        recommended_fused_DW=chosen=='candidate',fusion_consistently_faster_each_order=fused_consistently_faster if methods['methods']['candidate']['performance_eligible'] else None,
        entry='scripts/train_qwen3_vocab_v12.py',model='Qwen/Qwen3-8B',layers=36,world_size=4,
        batch_per_rank=1,sequence=512,hardware='RTX A6000 SM86',compute='BF16',master_gradient_moments='FP32',
        memory_cap_GiB=41.875,input_identity=manifest['input_identity'],comparisons=comparisons,
        extra_flags=dict(triton_optimized=flags+['--gemm-plan','results/qwen3_triton_v10_gemm_plan_001.json']),
        auxiliary_HF_port=dict(qualified=methods['methods']['HF_custom_head']['qualified'],
            performance_eligible=methods['methods']['HF_custom_head']['performance_eligible'],
            scope='Auxiliary experiment, not a recommended HF configuration when its gradient gate fails.'),
        measured_training_peak={f'r{o}':review['timing'][f'{chosen}_r{o}'] for o in [1,2]},
        recommendation_policy_sha256=sha(policy_path),review_sha256=sha(formal/'per_method_review.json'),
        whole_cohort_all_profiles_passed=review['qualified_training_improvement'],
        excluded_from_performance_claims=methods['excluded_from_performance_claims'],
        scope='Descriptor, not a CLI loader. Supply original protocol/input/full-quality/reference arguments. Loss-only labeled training/evaluation; diagnostic logits only outside timing. Head graph, ownership and collective schedule are customized. HF can port the same operator; native HF does not use that custom path. No inference/convergence claim.')
    with strategy_path.open('x') as f:f.write(json.dumps(strategy,indent=2)+'\n')
    index=dict(complete=True,qualified_training_improvement=True,default_changed=False,copies=copies,
        strategy=dict(path=strategy_path.name,sha256=sha(strategy_path)),raw_evidence_sha256=evidence,
        collector_sha256=sha(Path(__file__)),limitations=review['limitations'],
        original_whole_cohort_all_profiles_passed=review['qualified_training_improvement'],
        scope='Byte-identical small evidence copies. No weights, corpus, caches, raw gradient archives, logs, traces or source-archive payloads. Historical source paths/hashes and preserved Git history identify earlier screening implementations.')
    with index_path.open('x') as f:f.write(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(complete=True,recommended_arm=chosen,recommended_fused_DW=chosen=='candidate',comparisons=comparisons),indent=2))


if __name__=='__main__':main()
