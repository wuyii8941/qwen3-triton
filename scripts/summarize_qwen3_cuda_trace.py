"""Summarize an existing rank-local forward/backward trace, without GPU work.

Kernel duration sums may overlap. They are not optimizer-step fractions or
critical-path attribution. The V12 traces are captured after the 27 updates and
exclude AdamW; measured step timings must come from the full training report.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def union_us(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        else:
            total += max(0.0, stop - end)
        end = max(stop, end) if end is not None else stop
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(Path('/data1/tzh'))
    payload = args.trace.read_bytes()
    events = [e for e in json.loads(payload)['traceEvents'] if e.get('cat') == 'kernel']
    assert events and all(e['dur'] >= 0 for e in events)
    groups = defaultdict(list)
    for event in events:
        groups[event['name']].append((event['ts'], event['ts'] + event['dur']))
    rows = [dict(name=name, count=len(intervals), summed_seconds=sum(b-a for a,b in intervals)/1e6,
                 union_seconds=union_us(intervals)/1e6) for name, intervals in groups.items()]
    rows.sort(key=lambda row: row['summed_seconds'], reverse=True)
    intervals = [(e['ts'], e['ts']+e['dur']) for e in events]
    report = dict(complete=True, trace_sha256=hashlib.sha256(payload).hexdigest(),
                  source_trace=str(args.trace.resolve()), kernel_count=len(events),
                  kernel_span_seconds=(max(b for a,b in intervals)-min(a for a,b in intervals))/1e6,
                  any_kernel_active_seconds=union_us(intervals)/1e6,
                  kernel_summed_seconds=sum(e['dur'] for e in events)/1e6, kernels=rows,
                  scope='Rank0 post-update forward/backward only, no AdamW. Duration sums can overlap; '
                        'not whole-step fractions, speedup predictions, or critical-path attribution.')
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(kernel_count=len(events), top_kernels=rows[:8], scope=report['scope']), indent=2))


if __name__ == '__main__':
    main()
