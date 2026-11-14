"""Summarize an audited cohort using a policy frozen before timed updates.

This reports a profile-specific recommendation, leaving model defaults alone.
It verifies source/evidence hashes and recomputes step medians from raw ranks.
"""
from pathlib import Path
import argparse,hashlib,json,math,statistics


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();root=args.cohort.resolve()
    assert not args.destination.resolve().is_relative_to(Path('/home/tzh'))
    manifest_path=root/'manifest.json';review_path=root/'independent_review.json'
    manifest=json.loads(manifest_path.read_text());review=json.loads(review_path.read_text())
    policy=json.loads(args.policy.read_text())
    assert review['complete'] and review['audit_passed']
    assert policy['schema']==1 and policy['cohort']==root.name
    assert policy['manifest_sha256']==sha(manifest_path)
    assert policy['declared_before_any_measured_optimizer_update']
    for name,anchor in policy['log_prefix_anchors'].items():
        with (root/(name+'.log')).open('rb') as f:prefix=f.read(anchor['log_prefix_bytes'])
        assert hashlib.sha256(prefix).hexdigest()==anchor['log_prefix_sha256']
        for line in prefix.decode(errors='replace').splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            assert not (event.get('event')=='optimizer_update' and event.get('measured'))
    for record in (manifest,review):
        for path,h in record['sources'].items():assert sha(path)==h,path
    assert len(manifest['roles'])==8
    assert review['roles']==manifest['roles']
    timing={};qualified={};evidence={str(manifest_path):sha(manifest_path),str(review_path):sha(review_path),str(args.policy):sha(args.policy)}
    for role in manifest['roles']:
        name=role['name'];result_path=root/name/'result.json';monitor_path=root/(name+'.gpu.json')
        result=json.loads(result_path.read_text());monitor=json.loads(monitor_path.read_text())
        assert result['complete'] and result['model']=='Qwen/Qwen3-8B' and result['layers']==36 and result['world_size']==4
        assert result['input_manifest_sha256']==manifest['input_manifest_sha256']
        assert result['input_root']==manifest['input_root']
        assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and monitor['observation_errors']==0
        assert monitor['samples']
        owned=set(monitor['known_owned_pids'])
        for sample in monitor['samples']:
            assert not sample.get('foreign_pids') and 'observation_error' not in sample
            assert all(app['pid'] in owned for app in sample['apps'])
        samples=None
        for rank in result['ranks']:
            assert rank['updates']==27 and rank['parameter_count']==399
            current=[row['wall_seconds'] for row in rank['rows'] if row['measured']]
            assert len(current)==8 and all(math.isfinite(v) and v>0 for v in current)
            if samples is None:samples=current
            else:assert current==samples
        row=dict(median_seconds=statistics.median(samples),samples_seconds=samples,
            peak_allocated_gib=max(r['measured_training_peak']['allocated_gib'] for r in result['ranks']),
            peak_reserved_gib=max(r['measured_training_peak']['reserved_gib'] for r in result['ranks']))
        recorded=review['timing'][name]
        for key in row:assert row[key]==recorded[key],(name,key)
        timing[name]=row
        trace=review['traces'][name];trace_path=root/name/'training_trace.json'
        assert sha(trace_path)==trace['trace_sha256']
        trace_passed=bool(trace['native_Flash_kernels']) if role['arm']=='hf' else (trace['coverage_passed'] and trace['launch_geometry']['passed'])
        assert len(review['numerics'][name])==4
        qualified[name]=all(r['passed'] and r['parameter_count']==399 for r in review['numerics'][name]) and review['quality'][name]['passed'] and trace_passed
        for path in (result_path,monitor_path,trace_path):evidence[str(path)]=sha(path)
    controls_passed=all(qualified[name] for name in policy['controls'])
    candidates=[]
    for arm in policy['candidates']:
        plan_record=manifest['plans'][arm];assert sha(plan_record['path'])==plan_record['sha256']
        plan=json.loads(Path(plan_record['path']).read_text());evidence[plan_record['path']]=plan_record['sha256']
        orders=[]
        for order in policy['orders']:
            current=timing[f'{arm}_r{order}'];base=timing[f'v9_r{order}'];hf=timing[f'hf_r{order}']
            orders.append(dict(order=order,candidate_seconds=current['median_seconds'],V9_seconds=base['median_seconds'],HF_seconds=hf['median_seconds'],
                reduction_vs_V9=1-current['median_seconds']/base['median_seconds'],
                reduction_vs_HF=1-current['median_seconds']/hf['median_seconds'],
                peak_allocated_gib=current['peak_allocated_gib'],peak_reserved_gib=current['peak_reserved_gib'],
                extra_allocated_gib_vs_V9=current['peak_allocated_gib']-base['peak_allocated_gib']))
        gates=controls_passed and all(qualified[f'{arm}_r{o}'] for o in policy['orders'])
        performance=all(r['reduction_vs_V9']>=policy['minimum_latency_reduction_vs_V9_each_order'] for r in orders)
        memory=all(r['extra_allocated_gib_vs_V9']<=policy['maximum_extra_allocated_GiB_vs_V9'] and max(r['peak_allocated_gib'],r['peak_reserved_gib'])<=policy['memory_cap_GiB'] for r in orders)
        candidates.append(dict(arm=arm,plan=plan_record,entries=len(plan['entries']),orders=orders,
            numerical_quality_coverage_passed=gates,performance_rule_passed=performance,memory_rule_passed=memory,
            eligible=gates and performance and memory,
            observed_faster_than_HF_both_orders=all(r['reduction_vs_HF']>0 for r in orders)))
    eligible=[c for c in candidates if c['eligible']]
    winner=max(eligible,key=lambda c:(c['observed_faster_than_HF_both_orders'],min(r['reduction_vs_V9'] for r in c['orders']),-c['entries'])) if eligible else None
    report=dict(complete=True,audit_passed=True,model='Qwen/Qwen3-8B',layers=36,world_size=4,
        controls_passed=controls_passed,raw_all_acceptance_gates_passed=review['all_acceptance_gates_passed'],
        numerical_quality_coverage_by_role=qualified,timing=timing,candidates=candidates,
        recommended_candidate=winner['arm'] if winner else None,
        default_changed=False,convergence_claimed=False,inference_claimed=False,
        limitation='Two orders and eight measured updates per role are descriptive. No statistical significance or global-optimum claim. All gradients were compared at runtime; dense candidate gradients are not archived.',
        sources={**evidence,str(Path(__file__)):sha(Path(__file__))},scope=__doc__)
    with args.destination.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,recommended_candidate=report['recommended_candidate'],candidates=candidates),indent=2),flush=True)


if __name__=='__main__':main()
