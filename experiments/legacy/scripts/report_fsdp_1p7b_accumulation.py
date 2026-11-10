"""Fresh raw-file audit, only fixed globalbatch4 within this study."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.screen_fsdp_1p7b_accumulation import SPECS,audit_model


def pair_ratios(runs,faster,slower):
    return [runs[(r,slower)]['median_step_seconds']/runs[(r,faster)]['median_step_seconds']
        if (r,slower) in runs and (r,faster) in runs else None for r in (0,1)]


def summarize(root):
    raw_summary=(root/'result.json').read_bytes();summary=json.loads(raw_summary)
    expected=[(repeat,name) for repeat,order in ((0,list(SPECS)),(1,list(reversed(SPECS)))) for name in order]
    if (summary.get('complete') is not True or summary['specs']!={n:list(s) for n,s in SPECS.items()} or
            [(r['repeat'],r['case']) for r in summary['records']]!=expected):
        raise ValueError('Complete explicit ABC/CBA fixed-batch study required')
    signatures=set();runs={}
    for row in summary['records']:
        repeat,name=row['repeat'],row['case'];label=f'r{repeat}_{name}'
        log=(root/(label+'.log')).read_bytes()
        if hashlib.sha256(log).hexdigest()!=row['log_sha256']:raise ValueError('Changed worker log')
        if row['status']=='oom':
            if row['returncode']==0 or not any(t in log for t in
                    (b'CUDA out of memory',b'torch.OutOfMemoryError',b'torch.cuda.OutOfMemoryError')):
                raise ValueError('Invalid capacity failure')
            continue
        path=root/label/'result.json';raw=path.read_bytes();result=json.loads(raw)
        if (row['status']!='passed' or row['returncode']!=0 or
                Path(row['source']).resolve()!=path.resolve() or hashlib.sha256(raw).hexdigest()!=row['sha256']):
            raise ValueError('Result provenance mismatch')
        signatures.add(audit_model(result,SPECS[name]))
        if (result['manifest']['implementation_sources']!=summary['implementation_sources'] or
                result['manifest']['input_sha256_by_rank'][0]!=result['manifest']['input_sha256']):
            raise ValueError('Source or rank0/all-rank input identity mismatch')
        if (row['median_step_seconds']!=result['median_step_seconds'] or
                row['tokens_per_second']!=result['global_tokens_per_second'] or
                row['peak_allocated_gib']!=max(r['peak_allocated_gib'] for r in result['records'])):
            raise ValueError('Summary metrics differ from raw records')
        runs[(repeat,name)]=result
    if len(signatures)!=1 or any((r,'large_batch') not in runs for r in (0,1)):
        raise ValueError('Passing same-effective-batch controls required')
    comparisons={}
    for faster,slower in (('large_batch','accum_each'),('large_batch','accum_defer'),('accum_defer','accum_each')):
        ratios=pair_ratios(runs,faster,slower)
        comparisons[f'{faster}_over_{slower}']=dict(paired_throughput_ratios=ratios,
            over_two_percent_both_orders=all(v is not None and v>1.02 for v in ratios))
    cases={}
    for name in SPECS:
        records=[runs[(r,name)] for r in (0,1) if (r,name) in runs]
        differences=[]
        for repeat in (0,1):
            if (repeat,name) in runs:
                differences.append(max(abs(a['loss']-b['loss']) for a,b in zip(
                    runs[(repeat,name)]['records'],runs[(repeat,'large_batch')]['records'])))
        cases[name]=dict(spec=SPECS[name],median_seconds_by_order=[r['median_step_seconds'] for r in records],
            tokens_per_second_by_order=[r['global_tokens_per_second'] for r in records],
            peak_allocated_gib=max((r['peak_allocated_gib'] for r in summary['records']
                if r['case']==name and r['status']=='passed'),default=None),
            mean_loss_max_abs_vs_large_batch_by_order=differences)
    eligible=[n for n,r in cases.items() if len(r['median_seconds_by_order'])==2]
    winner=min(eligible,key=lambda n:sum(cases[n]['median_seconds_by_order']))
    return dict(complete=True,physical_gpus=[0,2],world_size=2,effective_global_batch=4,seq=512,
        cases=cases,comparisons=comparisons,fastest_observed_by_sum_of_medians=winner,
        input_identity_scope='all_ranks',raw_summary_path=str((root/'result.json').resolve()),
        raw_summary_sha256=hashlib.sha256(raw_summary).hexdigest(),
        implementation_sha256=summary['implementation_sources']['sha256'],
        frozen_content_sha256=summary['frozen_content_sha256'],defaults_changed=False,convergence_proven=False,
        scope='Real1.7B B2/A1 vs B1/A2 at same globalbatch4, short paired training steps. Different native BF16 shapes/accumulation may differ numerically; gate references match each microbatch partition. Not bitwise full-batch, long quality, significance or cross-globalbatch2 speedup proof')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=summarize(args.root)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
