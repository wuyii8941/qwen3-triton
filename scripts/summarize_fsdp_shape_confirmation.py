"""Explicit-workload two-order confirmation; no cross-shape speedup pooling."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.summarize_fsdp_memory import audit_result


def summarize(root, spec):
    root = Path(root)
    policies = spec['policies']; candidate = spec['candidate']
    if (len(policies) < 2 or candidate not in policies
            or any(not re.fullmatch(r'\w+', name) or len(p) not in (3,4)
                   for name, p in policies.items())):
        raise ValueError('Explicit distinct policies and candidate required')
    normalized=[tuple(p) if len(p)==4 else (*p,0) for p in policies.values()]
    if len(set(normalized))!=len(normalized) or any(type(p[3]) is not int or p[3]<0 for p in normalized):
        raise ValueError('Distinct effective policies and nonnegative CE chunk sizes required')
    first = list(policies)
    order = [f'r0_{p}' for p in first] + [f'r1_{p}' for p in reversed(first)]
    if (root/'status.log').read_text().splitlines() != [f'{p} returncode=0' for p in order]:
        raise ValueError('Successful complete reverse-order confirmation required')
    workload = spec['workload']
    required = {'model', 'world_size', 'batch', 'seq', 'warmup', 'steps',
                'memory_budget_gib', 'precision', 'attention'}
    if set(workload) != required or workload['warmup'] < 1 or workload['steps'] < 12:
        raise ValueError('Full explicit workload and at least 12 measured steps required')
    rows = []; signatures = set(); times = {}
    for name in order:
        policy = name[3:]; path = root/name/'result.json'; raw = path.read_bytes()
        result = json.loads(raw)
        if any(result[k] != v for k, v in workload.items()):
            raise ValueError('Result does not match prescribed workload')
        declared=policies[policy]
        if len(declared)==4 and result['optimization'].get('ce_only_chunk_size',0)!=declared[3]:
            raise ValueError('Mislabeled CE-only policy')
        signatures.add(audit_result(result, tuple(declared[:3]),allow_ce_only=len(declared)==4))
        times[name] = result['median_step_seconds']
        log = root/(name+'.log')
        rows.append(dict(case=name, source=str(path), sha256=hashlib.sha256(raw).hexdigest(),
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            median_step_seconds=result['median_step_seconds'],
            measured_records=[r for r in result['records'] if not r['warmup']],
            initial_hf_parity=result['initial_hf_parity']))
    if len(signatures) != 1:
        raise ValueError('Workload, source, inputs or allocator mismatch')
    pairs = [{p: times[f'r{r}_{p}']/times[f'r{r}_{candidate}']
              for p in policies if p != candidate} for r in (0, 1)]
    return dict(complete=True, spec=spec, records=rows,
        paired_candidate_throughput_ratios=pairs,
        candidate_over_two_percent_in_both_orders=all(v > 1.02 for p in pairs for v in p.values()),
        candidate_convergence_proven=False, workload_signature=list(next(iter(signatures))),
        scope='Fixed-workload reverse-order repetitions; not statistical significance or convergence')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    report = summarize(args.root, json.loads(args.spec.read_text()))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(report['paired_candidate_throughput_ratios']))


if __name__ == '__main__': main()
