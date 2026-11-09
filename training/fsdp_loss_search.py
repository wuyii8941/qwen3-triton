"""Validate a loss-only ablation; permit only its chunk-size to vary."""
import math
from training.fsdp_search import workload_signature


def summarize_loss_screen(rows):
    passing = []; signatures = set(); chunks = set(); retentions = set()
    for row in rows:
        chunk = row['chunk_size']
        if chunk in chunks or chunk < 0:
            raise ValueError('Duplicate/invalid chunk size')
        chunks.add(chunk)
        if row['status'] == 'oom':
            continue
        if row['status'] != 'passed':
            raise ValueError('Unresolved non-capacity error')
        r = row['result']
        if not r.get('complete') or r['optimization'].get('loss_chunk_size', 0) != chunk:
            raise ValueError('Incomplete/mislabeled loss result')
        # workload_signature's final element is the ONLY experimental variable.
        signatures.add(workload_signature(r)[:-1])
        retentions.add(r['optimization']['keep_full_layers'])
        time = r['median_step_seconds']
        parity = r.get('initial_hf_parity', {}).get('max_abs', float('inf'))
        peaks = [x['peak_allocated_gib'] for x in r['records']]
        if not (math.isfinite(time) and time > 0 and math.isfinite(parity) and 0 <= parity <= .03
                and peaks and all(math.isfinite(x) and 0 <= x <= r['memory_budget_gib'] for x in peaks)):
            raise ValueError('Invalid latency/parity/memory')
        passing.append(row)
    if len(signatures) != 1 or len(retentions) != 1 or not any(x['chunk_size'] == 0 for x in passing):
        raise ValueError('Matched native-loss baseline required')
    baseline = next(x for x in passing if x['chunk_size'] == 0)
    best = min(passing, key=lambda x: x['result']['median_step_seconds'])
    return dict(best_chunk_size=best['chunk_size'],
        throughput_speedup=baseline['result']['median_step_seconds']/best['result']['median_step_seconds'],
        scope='Fixed-workload loss screen only; not convergence or statistical significance')
