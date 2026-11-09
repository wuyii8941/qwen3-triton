"""Read-only convergence evidence reporting; partial runs never imply plateau."""
import argparse
import json
import math
from pathlib import Path
from training.convergence_rules import Plateau


def summarize(result):
    complete=bool(result.get('complete'))
    converged=bool(result.get('converged'))
    if converged and (not complete or result.get('stop_reason')!='validation_plateau'):
        raise ValueError('Invalid convergence claim')
    validation=result.get('validation',[])
    plateau_status=None
    rule=result.get('protocol',{}).get('plateau')
    if rule:
        plateau=Plateau(**rule)
        reached=False
        for row in validation:
            _,reached=plateau.update(row['loss'])
        plateau_status=dict(checks=plateau.checks,stale_checks=plateau.stale,
            patience=plateau.patience,reference_loss=plateau.best if validation else None,
            criterion_reached=reached)
    return dict(backend=result['backend'],complete=complete,converged=converged,
        stop_reason=result.get('stop_reason','running'),updates=len(result.get('training',[])),
        validation_checks=len(validation),initial_loss=validation[0]['loss'] if validation else None,
        best_observed_validation=min((x['loss'] for x in validation),default=None),
        latest_validation=validation[-1] if validation else None,
        plateau=plateau_status,
        heldout_test=result.get('test') if complete and converged else None,
        scope='Fixed WikiText task plateau, not original model pretraining convergence')


def compare_jobs(baseline,candidate):
    """Job-to-plateau comparison, explicitly NOT equal-quality speedup."""
    if not baseline.get('protocol') or baseline['protocol']!=candidate.get('protocol'):
        raise ValueError('Matching explicit protocols required')
    summaries=[summarize(r) for r in (baseline,candidate)]
    if not all(s['complete'] and s['converged'] for s in summaries):
        return dict(ready=False,reason='Both jobs must finish at validation plateau')
    if not all(s['plateau'] and s['plateau']['criterion_reached'] for s in summaries):
        raise ValueError('Validation history does not establish the declared plateau')
    for r in (baseline,candidate):
        if not r.get('best_parameters_restored') or not r.get('test'):
            raise ValueError('Best-parameter restoration and held-out test required')
        values=[r['elapsed_seconds'],r['best_validation'],r['test']['loss']]
        if not all(math.isfinite(v) for v in values) or values[0]<=0:
            raise ValueError('Invalid completed-job metrics')
    if any(baseline['test'][k]!=candidate['test'][k] for k in ('tokens','blocks')):
        raise ValueError('Different test coverage')
    return dict(ready=True,baseline=baseline['backend'],candidate=candidate['backend'],
        candidate_over_baseline_job_time=candidate['elapsed_seconds']/baseline['elapsed_seconds'],
        candidate_minus_baseline_best_validation=candidate['best_validation']-baseline['best_validation'],
        candidate_minus_baseline_test_loss=candidate['test']['loss']-baseline['test']['loss'],
        scope='Matched-protocol job-to-plateau runtime; not equal-quality speedup or statistical significance')


def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+')
    p.add_argument('--compare',action='store_true',help='Compare two completed plateau jobs in baseline/candidate order')
    a=p.parse_args()
    records=[json.loads(path.read_text()) for path in a.results]
    if any(r['protocol']!=records[0]['protocol'] for r in records):
        raise ValueError('Different convergence protocols; do not compare time-to-quality')
    if a.compare and len(records)!=2:p.error('--compare requires exactly two results')
    output=compare_jobs(*records) if a.compare else [summarize(r) for r in records]
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
