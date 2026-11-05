"""Report paired steady time, memory, numerical trajectory gate and startup."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    rows=[]
    for path in sorted(root.glob('r*_b*_s*_*.json')):
        row=json.loads(path.read_text())
        row['round']=int(path.name.split('_')[0][1:])
        rows.append(row)
    summaries=[]
    for batch,seq in sorted({(r['batch'],r['seq']) for r in rows}):
        for variant in ('candidate','multi','both'):
            pairs=[]
            for row in rows:
                if (row['batch'],row['seq'],row['stage'])!=(batch,seq,variant):continue
                baseline=next((b for b in rows if (b['batch'],b['seq'],b['stage'],b['round'])==(batch,seq,'baseline',row['round'])),None)
                if baseline is None:continue
                loss_abs=max(abs(a-b) for a,b in zip(row['losses'],baseline['losses']))
                pairs.append(dict(round=row['round'],baseline_ms=baseline['median_step_ms'],
                    candidate_ms=row['median_step_ms'],time_reduction=1-row['median_step_ms']/baseline['median_step_ms'],
                    peak_saved_mib=(baseline['peak_allocated_bytes']-row['peak_allocated_bytes'])/2**20,
                    max_training_loss_abs=loss_abs,trajectory_gate=loss_abs<=.01,
                    first_step_extra_seconds=(row['first_step_ms']-baseline['first_step_ms'])/1000))
            if pairs:
                summaries.append(dict(batch=batch,seq=seq,variant=variant,pairs=pairs,
                    median_time_reduction=statistics.median(p['time_reduction'] for p in pairs),
                    repeated_gain=len(pairs)>=2 and all(p['time_reduction']>.01 and p['trajectory_gate'] for p in pairs),
                    note='Single-round shapes remain provisional; timings are shared-host wall time, not convergence.'))
    progress=json.loads((root/'progress.json').read_text())
    result=dict(complete=progress.get('complete',False),completed_jobs=len(progress['jobs']),
                failed_jobs=[j for j in progress['jobs'] if j['status']=='failed'],summaries=summaries)
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    for row in summaries:
        print(f"B{row['batch']} S{row['seq']} {row['variant']}: "
              f"{row['median_time_reduction']*100:.2f}% time reduction, "
              f"{len(row['pairs'])} pairs, repeat_gate={row['repeated_gain']}")
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('artifacts/optimization_matrix_v1'))
    args=parser.parse_args();summarize(args.root)
