"""Read-only audit of live or finished convergence jobs; never infers success.

The main-entry clock excludes Python imports/teardown. External observer
times include them, but event timestamps have polling/scheduling uncertainty.
"""
import argparse
import json
from pathlib import Path


def events(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A concurrent writer may not yet have finished its last line.
                continue
    return rows


def audit(root):
    protocol = json.loads((root / 'frozen_protocol.json').read_text())
    jobs = {}
    curves = {}
    for name in ('native', 'triton'):
        directory = root / name
        rows = events(directory / 'events.jsonl')
        curves[name] = [r for r in rows if r.get('event') in ('initial_validation', 'validation')]
        summary_path = directory / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        jobs[name] = dict(
            summary=summary,
            latest_event=rows[-1] if rows else None,
            validation_checks=sum(r.get('event') == 'validation' for r in rows),
            checkpoint_seconds=sum(r['seconds'] for r in rows if r.get('event') == 'checkpoint'),
            # Validation duration is not separately instrumented in frozen jobs.
            validation_seconds=None,
        )
    finished = all(j['summary'] is not None for j in jobs.values())
    converged = finished and all(j['summary']['status'] == 'converged' for j in jobs.values())
    report = dict(jobs=jobs, finished=finished, converged_both=converged,
                  shared_preprocessing_seconds=None,
                  limitation='Shared preprocessing unmeasured; one sequential job pair, not repeated trials.')
    if not finished:
        return report
    target = min(r['loss'] for r in curves['native']) + protocol['plateau']['min_delta']
    crossings = {}
    for name, curve in curves.items():
        crossing = next((r for r in curve if r['loss'] <= target), None)
        live = events(root / name / 'external_clock_events.jsonl')
        observed = next((r for r in live if crossing is not None
                         and r.get('job_event', {}).get('event') == crossing['event']
                         and r['job_event'].get('step') == crossing['step']), None)
        crossings[name] = dict(job_event=crossing,
                              external_observed_seconds=observed['process_elapsed_seconds'] if observed else None)
    report.update(quality_target=target, first_crossings=crossings)
    # Ratios are withheld for an incomplete or safety-limited comparison.
    if converged:
        report['main_entry_plateau_speedup'] = jobs['native']['summary']['elapsed_seconds'] / jobs['triton']['summary']['elapsed_seconds']
        if all(c['job_event'] is not None for c in crossings.values()):
            report['main_entry_matched_quality_speedup'] = crossings['native']['job_event']['elapsed_seconds'] / crossings['triton']['job_event']['elapsed_seconds']
        if all(c['external_observed_seconds'] is not None for c in crossings.values()):
            report['external_observed_matched_quality_speedup'] = crossings['native']['external_observed_seconds'] / crossings['triton']['external_observed_seconds']
        clocks_path = root / 'external_process_clocks.json'
        clocks = json.loads(clocks_path.read_text()) if clocks_path.exists() else {}
        if all(n in clocks and clocks[n]['summary_present'] for n in jobs):
            report['external_process_lifetime_speedup'] = clocks['native']['process_lifetime_seconds'] / clocks['triton']['process_lifetime_seconds']
            report['external_clocks'] = clocks
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().root), indent=2))
