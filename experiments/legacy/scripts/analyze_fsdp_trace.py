"""Summarize GPU interval unions, not additive predictions of step latency."""
import argparse
import json
import hashlib
import math
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def summarize(trace):
    devices = {}
    for event in trace.get('traceEvents', []):
        if event.get('cat') != 'kernel' or event.get('ph') != 'X':
            continue
        start, length = event.get('ts'), event.get('dur')
        if not isinstance(start, (int, float)) or not isinstance(length, (int, float)):
            continue
        if not math.isfinite(start) or not math.isfinite(length):
            raise ValueError('Nonfinite GPU kernel interval')
        if length <= 0:
            continue
        # Never intersect kernels from distinct GPUs. Missing device identity
        # is rejected instead of silently mixing timelines by stream ID.
        device = event.get('args', {}).get('device')
        if device is None:
            raise ValueError('GPU kernel is missing args.device')
        row = devices.setdefault(str(device), {'communication': [], 'local': [], 'groups': {}})
        name = event.get('name', '')
        lowered = name.lower()
        communication = 'nccl' in lowered
        kind = 'communication' if communication else 'local'
        row[kind].append((start, start + length))
        group = row['groups'].setdefault(name, {'launches': 0, 'summed_kernel_us': 0.0})
        group['launches'] += 1
        group['summed_kernel_us'] += length
    if not devices:
        raise ValueError('No complete GPU kernel events found')
    output = {}
    for device, row in devices.items():
        communication, local = merge(row['communication']), merge(row['local'])
        overlap = intersection(communication, local)
        combined = merge(communication + local)
        output[device] = {
            'communication_union_us': duration(communication),
            'local_union_us': duration(local),
            'communication_local_overlap_us': overlap,
            'communication_without_local_us': duration(communication) - overlap,
            'gpu_active_union_us': duration(combined),
            'observed_kernel_span_us': combined[-1][1] - combined[0][0],
            'kernels': sorted((dict(name=name, **group) for name, group in row['groups'].items()),
                              key=lambda group: group['summed_kernel_us'], reverse=True),
        }
    return {'devices': output, 'scope': 'Observed GPU kernel intervals, Chrome trace microseconds',
            'caveats': ['NCCL duration includes waiting; it is not wire transfer time.',
                        'Non-overlapped communication is not a causal removable bottleneck.',
                        'Kernel span excludes CPU work outside the first/last GPU kernel.',
                        'Profiled timing is not unprofiled steady-state latency.',
                        'Local kernels include optimizer/recomputation; they are not only model compute.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    report = summarize(json.loads(raw))
    report['source'] = str(args.trace)
    report['source_sha256'] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(report, indent=2)
    if args.output:
        # Preserve previous experimental evidence rather than overwrite it.
        with args.output.open('x') as handle:
            handle.write(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
