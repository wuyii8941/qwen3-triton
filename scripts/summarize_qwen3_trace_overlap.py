"""Describe overlap in untimed complete-model forward/backward CUDA traces.

This is a diagnostic, not optimizer-step timing or transport-cost attribution.
NCCL kernels can wait on dependencies; category unions can overlap.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

from review_qwen3_triton_suite import trace_report


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval_report(events, model_names):
    sums = dict.fromkeys(('model', 'NCCL', 'framework'), 0.0)
    points = []
    for event in events:
        start, duration = event['ts'], event['dur']
        assert math.isfinite(start) and math.isfinite(duration) and duration > 0
        name = event['name']
        kind = 'NCCL' if name.startswith('nccl') else 'model' if name in model_names else 'framework'
        sums[kind] += duration
        points.extend(((start, kind, 1), (start + duration, kind, -1)))
    assert points
    points.sort()
    active = dict.fromkeys(sums, 0)
    totals = dict.fromkeys(('GPU_busy', 'GPU_idle', 'model', 'NCCL', 'framework',
                            'NCCL_without_model', 'NCCL_only', 'model_and_NCCL'), 0.0)
    last = points[0][0]
    for timestamp, kind, change in points:
        elapsed = timestamp - last
        on = {key for key, count in active.items() if count}
        totals['GPU_busy' if on else 'GPU_idle'] += elapsed
        for key in on:
            totals[key] += elapsed
        if 'NCCL' in on and 'model' not in on:
            totals['NCCL_without_model'] += elapsed
        if on == {'NCCL'}:
            totals['NCCL_only'] += elapsed
        if 'NCCL' in on and 'model' in on:
            totals['model_and_NCCL'] += elapsed
        active[kind] += change
        assert active[kind] >= 0
        last = timestamp
    assert not any(active.values())
    window = points[-1][0] - points[0][0]
    assert math.isclose(totals['GPU_busy'] + totals['GPU_idle'], window, abs_tol=1.e-5)
    assert all(0 <= value <= window + 1.e-5 for value in totals.values())
    return dict(profiled_GPU_window_seconds=window / 1.e6,
                union_seconds={key: value / 1.e6 for key, value in totals.items()},
                kernel_duration_sums_seconds={key: value / 1.e6 for key, value in sums.items()},
                event_count=len(events))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--role', action='append', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    assert not args.destination.resolve().is_relative_to(Path('/home/tzh'))
    root = args.cohort.resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    roles = {role['name']: role for role in manifest['roles']}
    reports = {}
    for name in args.role:
        role = roles[name]
        assert role['variant'] == 'triton_optimized'
        monitor_path = root / (name + '.gpu.json')
        monitor = json.loads(monitor_path.read_text())
        assert monitor['returncode'] == 0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
        path = root / name / 'training_trace.json'
        coverage = trace_report(path, role['variant'])
        assert coverage['coverage_passed']
        data = json.loads(path.read_text())
        events = [event for event in data['traceEvents'] if event.get('ph') == 'X'
                  and event.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and event.get('dur', 0) > 0]
        reports[name] = dict(trace_sha256=sha(path), monitor_sha256=sha(monitor_path),
                             **interval_report(events, set(coverage['observed_expected_kernels'])))
    output = dict(complete=True, roles=reports, scope=__doc__,
                  limitation='Rank0 untimed post-update27 forward/backward only; excludes optimizer. NCCL durations include possible dependency waits, not just transport. Category unions overlap; do not add them to estimate a step.',
                  sources={str(manifest_path): sha(manifest_path), str(Path(__file__)): sha(__file__),
                           str(Path(trace_report.__code__.co_filename)): sha(trace_report.__code__.co_filename)})
    with args.destination.open('x') as stream:
        stream.write(json.dumps(output, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
