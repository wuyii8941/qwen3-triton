"""Summarize initial screens, keeping LR search cost and failures visible."""
import argparse
import json
from pathlib import Path


def summarize(root):
    status=json.loads((root/'status.json').read_text())
    rows=[]
    for path in root.glob('train_*.json'):
        content=json.loads(path.read_text())
        if not isinstance(content,dict) or 'result' not in content:continue
        row=content['result']
        rows.append(dict(variant=row['variant'],lr=row['lr'],step_ms=row['median_step_ms'],
            peak_gib=row['peak_allocated_bytes']/1024**3,state_gib=row['optimizer_state_bytes']/1024**3,
            initial_validation=row['initial_validation'],final_validation=row['validation'][-1]['loss'],
            best_observed_validation=min(r['loss'] for r in row['validation']),
            target=row['target'],first_target_step=row['first_crossing']['step'] if row['first_crossing'] else None,
            elapsed_seconds=row['elapsed_seconds']))
    best={}
    for variant in ('native','flashoptim','muon','normuon','liger'):
        candidates=[r for r in rows if r['variant']==variant]
        if candidates:
            best[variant]=min(candidates,key=lambda r:r['best_observed_validation'])
    search_cost={variant:sum(s['wall_seconds'] for s in status['stages'] if s['name'].startswith('train_'+variant+'_'))
                 for variant in best}
    return dict(complete=status['complete'],trials=rows,best_validation_trial=best,
        lr_search_process_seconds=search_cost,failed_stages=[s for s in status['stages'] if s['returncode']],
        caveats=['64-step screen, not convergence or independent test-set quality.',
                 'Only16 validation blocks; first evaluation at8 steps limits time-to-target resolution.',
                 'Muon/NorMuon auxiliary AdamW LR fixed at1e-5; not a full joint LR search.',
                 'Shared GPU; small timing differences require further isolation.',
                 'Torch2.10 matched baseline; cannot mix with historical Torch2.5 results.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    report=summarize(args.root)
    (args.root/'initial_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
