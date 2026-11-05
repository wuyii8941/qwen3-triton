"""Partial or terminal evidence for three-way jobs; no invented speedups/CI."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from benchmarks.report_convergence_comparison import events


def summarize(root):
    plan = json.loads((root / 'plan.json').read_text())
    jobs = {}
    groups = {}
    for job in plan['jobs']:
        name = f's{job["seq"]}_seed{job["seed"]}_{job["backend"]}'
        directory = root / name
        rows = events(directory / 'events.jsonl')
        clock = events(root / f'{name}.clock.jsonl')
        summary_path = directory / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        protocol = root / f'{name}.protocol.json'
        if summary and summary['protocol_hash'] != hashlib.sha256(protocol.read_bytes()).hexdigest():
            raise ValueError(f'Protocol mismatch: {name}')
        exit_row = next((r for r in reversed(clock) if r['event'] == 'exit'), None)
        jobs[name] = dict(**job, summary=summary,
            latest_event=rows[-1] if rows else None,
            process_wall_seconds=exit_row['wall_seconds'] if exit_row else None,
            returncode=exit_row['returncode'] if exit_row else None,
            checkpoint_seconds=sum(r['seconds'] for r in rows if r['event'] == 'checkpoint'))
        groups.setdefault((job['seq'], job['seed']), {})[job['backend']] = (name, rows, clock)
    comparisons = []
    recovery_path=root/'recovery.json'
    recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
    for (seq, seed), backends in groups.items():
        if 'native' not in backends:
            continue
        native_name, native_rows, _ = backends['native']
        if native_name in recovery.get('timing_invalid_jobs',[]):
            # Interrupted/resumed runs are quality evidence, not an uninterrupted timing trial.
            continue
        native_summary = jobs[native_name]['summary']
        if not native_summary or native_summary['status'] != 'converged':
            continue
        protocol = json.loads((root / f'{native_name}.protocol.json').read_text())
        native_curve = [r for r in native_rows if r['event'] in ('initial_validation', 'validation')]
        target = min(r['loss'] for r in native_curve) + protocol['plateau']['min_delta']
        reached = {}
        for backend, (name, rows, clock) in backends.items():
            curve = [r for r in rows if r['event'] in ('initial_validation', 'validation')]
            first = next((r for r in curve if r['loss'] <= target), None)
            observed = next((r for r in clock if r['event'] == 'observed' and first is not None
                             and r['job_event']['event'] == first['event']
                             and r['job_event']['step'] == first['step']), None)
            reached[backend] = dict(step=first['step'] if first else None,
                loss=first['loss'] if first else None,
                observed_seconds=observed['process_elapsed_seconds'] if observed else None)
        for candidate in ('compile', 'triton'):
            if candidate not in backends:
                continue
            candidate_name = backends[candidate][0]
            pair = dict(seq=seq, seed=seed, candidate=candidate, quality_target=target,
                        crossings={'native': reached['native'], candidate: reached[candidate]})
            ns, cs = jobs[native_name], jobs[candidate_name]
            complete = all(x['summary'] and x['summary']['status'] == 'converged'
                           and x['returncode'] == 0 and x['process_wall_seconds'] is not None for x in (ns, cs))
            pair['converged_pair'] = complete
            if complete:
                pair['complete_job_speedup_vs_native'] = ns['process_wall_seconds'] / cs['process_wall_seconds']
                pair['selected_test_loss_delta'] = cs['summary']['test_loss'] - ns['summary']['test_loss']
                if all(x['observed_seconds'] is not None for x in pair['crossings'].values()):
                    pair['matched_quality_speedup_vs_native'] = reached['native']['observed_seconds'] / reached[candidate]['observed_seconds']
            comparisons.append(pair)
        # Direct Triton-vs-compile matters more than only beating eager.
        if all(b in backends for b in ('triton', 'compile')):
            cn, tn = backends['compile'][0], backends['triton'][0]
            complete = all(jobs[n]['summary'] and jobs[n]['summary']['status']=='converged'
                           and jobs[n]['returncode']==0 for n in (cn, tn))
            pair = dict(seq=seq, seed=seed, candidate='triton_vs_compile', converged_pair=complete)
            if complete:
                pair['complete_job_speedup'] = jobs[cn]['process_wall_seconds'] / jobs[tn]['process_wall_seconds']
                if all(reached[b]['observed_seconds'] is not None for b in ('compile', 'triton')):
                    pair['matched_quality_speedup'] = reached['compile']['observed_seconds'] / reached['triton']['observed_seconds']
            comparisons.append(pair)
    aggregate = []
    for seq in (512, 2048):
        for backend in ('compile', 'triton', 'triton_vs_compile'):
            metric = 'matched_quality_speedup' if backend=='triton_vs_compile' else 'matched_quality_speedup_vs_native'
            values = [p[metric] for p in comparisons if p['seq']==seq and p['candidate']==backend and metric in p]
            expected = sum(j['seq']==seq and j['backend']=='native' for j in plan['jobs'])
            aggregate.append(dict(seq=seq, candidate=backend, n=len(values), expected_seeds=expected,
                complete=len(values)==expected,
                geometric_mean=math.exp(statistics.mean(map(math.log,values))) if values else None,
                median=statistics.median(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None))
    return dict(jobs=jobs, comparisons=comparisons, aggregate=aggregate,recovery=recovery,
        all_jobs_terminated=all(j['returncode'] is not None for j in jobs.values()),
        all_jobs_converged=all(j['summary'] and j['summary']['status']=='converged' and j['returncode']==0
                               for j in jobs.values()),
        statistical_scope='S512 three paired data-order seeds; S2048 single-seed extension. Ranges are not confidence intervals.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.root)
    temp=args.root/'summary.tmp';temp.write_text(json.dumps(result,indent=2))
    temp.replace(args.root/'summary.json')
    print(json.dumps(dict(all_jobs_terminated=result['all_jobs_terminated'],
                         all_jobs_converged=result['all_jobs_converged'],aggregate=result['aggregate']),indent=2))
