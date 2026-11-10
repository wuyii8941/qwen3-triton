"""Fresh raw-file audit for the separate real1.7B two-card workload."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.screen_fsdp_1p7b import POLICIES
from scripts.summarize_fsdp_memory import audit_result
from scripts.audit_fsdp_exchange_confirmation import loss_comparison


def summarize(root):
    summary=json.loads((root/'result.json').read_text())
    if not summary.get('complete') or summary['physical_gpus']!=[0,2]:
        raise ValueError('Complete declared two-card experiment required')
    rows=summary['records']
    expected=[(repeat,name) for repeat,order in ((0,list(POLICIES)),(1,list(reversed(POLICIES))))
              for name in order]
    if [(r['repeat'],r['case']) for r in rows]!=expected:raise ValueError('ABCD/DCBA incomplete')
    signatures=set();runs={}
    for row in rows:
        repeat,name=row['repeat'],row['case'];label=f'r{repeat}_{name}'
        log=(root/(label+'.log')).read_bytes()
        if hashlib.sha256(log).hexdigest()!=row['log_sha256']:raise ValueError('Changed log')
        if row['status']=='oom':
            if row['returncode']==0 or not any(t in log for t in
                    (b'CUDA out of memory',b'torch.OutOfMemoryError',b'torch.cuda.OutOfMemoryError')):
                raise ValueError('Invalid capacity failure')
            continue
        if row['status']!='passed' or row['returncode']!=0:raise ValueError('Unexpected failure')
        path=root/label/'result.json';raw=path.read_bytes();result=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=row['sha256'] or Path(row['source']).resolve()!=path.resolve():
            raise ValueError('Raw result provenance mismatch')
        manifest=result['manifest']
        signature=audit_result(result,POLICIES[name])
        if (result['global_parameters']!=1720574976 or result['world_size']!=2 or
                result['batch']!=1 or result['seq']!=512 or result['warmup']!=3 or
                result['steps']!=12 or result['memory_budget_gib']!=24):
            raise ValueError('Different model/workload')
        hashes=manifest['input_sha256_by_rank']
        if len(hashes)!=2 or any(len(h)!=64 for h in hashes):raise ValueError('Both input hashes required')
        signatures.add(signature+(tuple(hashes),))
        if manifest['implementation_sources']!=summary['implementation_sources']:
            raise ValueError('Worker/supervisor source mismatch')
        if any(result['communication_environment'].values()) or any(result['exchange_callback_counts_by_rank']):
            raise ValueError('Unexpected protocol or custom exchange')
        if (row['median_step_seconds']!=result['median_step_seconds'] or
                row['peak_allocated_gib']!=max(r['peak_allocated_gib'] for r in result['records'])):
            raise ValueError('Summary metrics do not match raw records')
        runs[(repeat,name)]=result
    if len(signatures)!=1 or any((r,'native_k0') not in runs for r in (0,1)):
        raise ValueError('Matched passing native controls required')
    policies={}
    for name in POLICIES:
        ratios=[];loss_checks=[];peaks=[];medians=[]
        for repeat in (0,1):
            if (repeat,name) not in runs:continue
            result=runs[(repeat,name)];base=runs[(repeat,'native_k0')]
            ratios.append(base['median_step_seconds']/result['median_step_seconds'])
            loss_checks.append(loss_comparison(result,base))
            peaks.append(max(r['peak_allocated_gib'] for r in result['records']))
            medians.append(result['median_step_seconds'])
        policies[name]=dict(paired_throughput_ratios=ratios,median_seconds_by_order=medians,
            peak_allocated_gib=max(peaks) if peaks else None,loss_checks=loss_checks,
            candidate=len(ratios)==2 and all(math.isfinite(v) and v>1.02 for v in ratios) and
                all(c['within_declared_loss_tolerance'] for c in loss_checks))
    performance_eligible=[name for name,row in policies.items()
        if len(row['paired_throughput_ratios'])==2 and all(v>1.02 for v in row['paired_throughput_ratios'])]
    performance_winner=min(performance_eligible,key=lambda n:sum(policies[n]['median_seconds_by_order'])) \
        if performance_eligible else 'native_k0'
    eligible=[name for name,row in policies.items() if row['candidate']]
    selected=min(eligible,key=lambda n:sum(policies[n]['median_seconds_by_order'])) if eligible else 'native_k0'
    repeat_check=loss_comparison(runs[(1,'native_k0')],runs[(0,'native_k0')])
    checkpoint_ratios=[runs[(r,'native_k28')]['median_step_seconds']/
        runs[(r,'checkpoint_k28')]['median_step_seconds'] for r in (0,1)] \
        if all((r,n) in runs for r in (0,1) for n in ('native_k28','checkpoint_k28')) else []
    return dict(complete=True,policies=policies,performance_screening_winner=performance_winner,
        native_repeat_loss_check=repeat_check,
        native_repeat_passes_short_loss_tolerance=repeat_check['within_declared_loss_tolerance'],
        checkpoint_vs_full_retention_ratios=checkpoint_ratios,
        checkpoint_beats_full_retention_both_orders=len(checkpoint_ratios)==2 and all(v>1.02 for v in checkpoint_ratios),
        candidate_passing_short_loss_gate=selected if eligible else None,
        screening_winner=selected,
        frozen_content_sha256=summary['frozen_content_sha256'],implementation_sources=summary['implementation_sources'],
        raw_summary_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
        defaults_changed=False,convergence_proven=False,
        scope='Real1.7B world2 GPU0/2 B1 S512 globalbatch2 24GiB, two ordered short runs; not 8B/scaling or long-quality proof')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=summarize(args.root)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
