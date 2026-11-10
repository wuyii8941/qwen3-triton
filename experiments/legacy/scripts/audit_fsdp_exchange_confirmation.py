"""Independently re-read raw model runs, not just the supervisor's decision."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.confirm_fsdp_exchange import CASES,audit_case,decisions
from scripts.fsdp_resource_budget import budget_ready
from scripts.queue_nccl_then_quality import resources_ready


def audit_launch(root,label,policy):
    if (policy.get('allocator_budget_gib')!=42. or type(policy.get('idle_confirmations')) is not int
            or policy['idle_confirmations']<1):
        raise ValueError('Explicit launch budget and confirmations required')
    path=root/(label+'_resources.json');raw=path.read_bytes();record=json.loads(raw)
    for key in ('allocator_budget_gib','free_margin_gib','idle_confirmations'):
        if record.get(key)!=policy.get(key):raise ValueError('Resource record policy mismatch')
    samples=record['samples']
    if len(samples)!=policy['idle_confirmations']:raise ValueError('Incomplete consecutive idle samples')
    previous=None
    for sample in samples:
        tick=sample['unix_seconds'];readings=sample['readings']
        if (not isinstance(tick,(int,float)) or not math.isfinite(tick) or tick<=0 or
                (previous is not None and tick<=previous) or len(readings)!=4 or
                any(len(row)!=3 or any(type(v) is not int or v<0 for v in row) for row in readings)):
            raise ValueError('Invalid ordered resource observations')
        previous=tick
        margin=policy.get('free_margin_gib')
        valid=(resources_ready(readings) if margin is None else
            budget_ready(readings,allocator_gib=42.,free_margin_gib=margin))
        if not valid:raise ValueError('Launch did not satisfy declared memory/utilization guard')
    return hashlib.sha256(raw).hexdigest()


def callback_contract(result,spec):
    backend,scope=spec[1:3]
    expected=({f'model.layers.{i}' for i in range(36)} if scope=='decoder'
              else {'lm_head'} if scope=='head' else set()) if backend else set()
    for counts in result['exchange_callback_counts_by_rank']:
        if set(counts)!=expected or any(v!=15 for v in counts.values()):
            raise ValueError('Wrong selected units or callback count; no hidden fallback allowed')


def loss_comparison(candidate,baseline,*,atol=1e-4,rtol=1e-5):
    a=candidate['records'];b=baseline['records']
    if len(a)!=len(b):raise ValueError('Loss history length mismatch')
    differences=[abs(x['loss']-y['loss']) for x,y in zip(a,b)]
    return dict(max_absolute_loss_difference=max(differences),atol=atol,rtol=rtol,
        within_declared_loss_tolerance=all(math.isfinite(d) and d<=atol+rtol*abs(y['loss'])
            for d,y in zip(differences,b)),
        scope='Recorded mean loss trajectory only, not full gradient/moment or convergence equivalence')


def audit(root):
    summary=json.loads((root/'result.json').read_text())
    if summary.get('complete') is not True:raise ValueError('Completed supervisor required')
    expected=[(repeat,name) for repeat,order in ((0,list(CASES)),(1,list(reversed(CASES))))
              for name in order]
    rows=summary['records']
    if [(r['repeat'],r['case']) for r in rows]!=expected:
        raise ValueError('Complete ABCD/DCBA order required')
    raw_runs={};signatures=set();resource_hashes={}
    for row in rows:
        repeat,name=row['repeat'],row['case'];label=f'r{repeat}_{name}'
        if 'resource_policy' in summary:
            resource_hashes[label]=audit_launch(root,label,summary['resource_policy'])
        log=(root/(label+'.log')).read_bytes()
        if hashlib.sha256(log).hexdigest()!=row['log_sha256']:
            raise ValueError('Log bytes changed')
        if row['status']=='oom':
            if row['returncode']==0 or not any(token in log for token in
                    (b'CUDA out of memory',b'torch.OutOfMemoryError',b'torch.cuda.OutOfMemoryError')):
                raise ValueError('Invalid capacity failure')
            continue
        if row['status']!='passed' or row['returncode']!=0:
            raise ValueError('Non-capacity failure is not a valid comparison')
        path=root/label/'result.json';raw=path.read_bytes();result=json.loads(raw)
        if (hashlib.sha256(raw).hexdigest()!=row['sha256'] or
                Path(row['source']).resolve()!=path.resolve()):
            raise ValueError('Result provenance mismatch')
        signatures.add(audit_case(result,CASES[name]));callback_contract(result,CASES[name])
        if result['manifest']['implementation_sources']!=summary['implementation_sources']:
            raise ValueError('Supervisor and worker source snapshots differ')
        for filename,digest in result['manifest']['exchange_sources'].items():
            if summary['helper_sources'].get(filename)!=digest:
                raise ValueError('Communication helper provenance mismatch')
        if (result['median_step_seconds']!=row['median_step_seconds'] or
                max(r['peak_allocated_gib'] for r in result['records'])!=row['peak_allocated_gib'] or
                max(r['peak_reserved_gib'] for r in result['records'])!=row['peak_reserved_gib']):
            raise ValueError('Summary metrics differ from raw worker')
        raw_runs[(repeat,name)]=result
    if len(signatures)!=1:raise ValueError('Mismatched complete model runs')
    for repeat in (0,1):
        for control in ('native_auto_k36','native_simple_k36'):
            if (repeat,control) not in raw_runs:raise ValueError('Both passing native controls required')
    calculated=decisions(rows)
    if calculated!=summary['decisions']:raise ValueError('Decision differs from raw audited metrics')
    trajectories={}
    for repeat in (0,1):
        baseline=raw_runs[(repeat,'native_auto_k36')]
        for name in CASES:
            if (repeat,name) in raw_runs and name!='native_auto_k36':
                trajectories[f'r{repeat}_{name}']=loss_comparison(raw_runs[(repeat,name)],baseline)
    # Stock NCCL protocol tuning is a valid model candidate too. Do not
    # discard a faster native control merely because it is not Triton.
    protocol_ratios=[raw_runs[(r,'native_auto_k36')]['median_step_seconds']/
        raw_runs[(r,'native_simple_k36')]['median_step_seconds'] for r in (0,1)]
    protocol_candidate=all(v>1.02 for v in protocol_ratios)
    candidates=[name for name,row in calculated.items()
        if row['over_two_percent_both_orders'] and all(
            trajectories[f'r{r}_{name}']['within_declared_loss_tolerance'] for r in (0,1))]
    if protocol_candidate and all(trajectories[f'r{r}_native_simple_k36']['within_declared_loss_tolerance']
            for r in (0,1)):
        candidates.append('native_simple_k36')
    return dict(complete=True,raw_runs_audited=len(raw_runs),decisions=calculated,
        native_simple_vs_native_auto=dict(paired_throughput_ratios=protocol_ratios,
            over_two_percent_both_orders=protocol_candidate),
        launch_resource_guard_audited=bool(resource_hashes),launch_resource_sha256=resource_hashes,
        loss_trajectory_checks=trajectories,
        candidates_for_quality=candidates,
        defaults_changed=False,convergence_proven=False,
        scope='Raw-file/hash/ordered-workload/callback/short-loss audit; not an independent training replay or significance proof')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.root)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
