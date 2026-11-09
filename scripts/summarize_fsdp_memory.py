"""Audit a completed activation/retention screen; never treat OOM as speed."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


CASES = {
    'native_k24': (24, 0, 'full'),
    'full_k24': (24, 2, 'full'),
    'selective_k24': (24, 2, 'selective'),
    'full_k32': (32, 2, 'full'),
    'full_k36': (36, 2, 'full'),
}


def audit_result(result, policy, *, vary_allocator=False, allow_ce_only=False, allow_exchange=False):
    if not result.get('complete') or result['backend'] != 'official':
        raise ValueError('Complete official worker required')
    opt = result['optimization']
    if opt.get('exchange_reduce') and not allow_exchange:
        raise ValueError('Exchange experiments require a dedicated explicit comparison')
    if opt.get('ce_only_chunk_size',0) and not allow_ce_only:
        raise ValueError('CE-only experiments require a dedicated explicit comparison')
    if tuple(opt[k] for k in ('keep_full_layers', 'checkpoint_every', 'checkpoint_mode')) != policy:
        raise ValueError('Mislabeled policy')
    if (result.get('accumulation', 1) != 1 or result.get('gradient_sync', 'each') != 'each'
            or opt.get('loss_chunk_size', 0) != 0 or opt.get('swiglu', 'native') != 'native'):
        raise ValueError('Unintended additional optimization')
    manifest = result['manifest']
    snapshot = manifest.get('implementation_sources', {}).get('sha256')
    if not snapshot or not manifest.get('input_sha256'):
        raise ValueError('Implementation and input hashes required')
    signature = tuple(result[k] for k in ('model', 'world_size', 'batch', 'seq', 'warmup',
        'steps', 'precision', 'attention', 'memory_budget_gib')) + (
            manifest['input_sha256'], snapshot, manifest['pytorch'], manifest['cuda'], manifest['gpu'],
            None if vary_allocator else json.dumps(result.get('allocator',{}),sort_keys=True))
    parity = result.get('initial_hf_parity', {}).get('max_abs', float('inf'))
    if not math.isfinite(parity) or not 0 <= parity <= .03:
        raise ValueError('Missing/invalid HF parity')
    records = result['records']
    warmup, steps = result['warmup'], result['steps']
    if warmup < 1 or steps < 1 or len(records) != warmup+steps:
        raise ValueError('Incomplete measured steps')
    for index, row in enumerate(records):
        if row['step'] != index+1 or row['warmup'] != (index < warmup):
            raise ValueError('Incorrect step/warmup sequence')
        if not all(math.isfinite(row[k]) for k in ('max_rank_seconds', 'loss',
                'peak_allocated_gib', 'peak_reserved_gib')):
            raise ValueError('Nonfinite measurement')
        if not (row['max_rank_seconds'] > 0 and 0 <= row['peak_allocated_gib'] <= result['memory_budget_gib']
                and row['peak_allocated_gib'] <= row['peak_reserved_gib'] <= result['memory_budget_gib']):
            raise ValueError('Invalid latency or memory')
    median = statistics.median(row['max_rank_seconds'] for row in records if not row['warmup'])
    if not math.isclose(median, result['median_step_seconds'], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('Reported median inconsistent with raw steps')
    expected_tps = result['world_size']*result['batch']*result['seq']/median
    if not math.isclose(expected_tps, result['global_tokens_per_second'], rel_tol=1e-10):
        raise ValueError('Throughput inconsistent with fixed workload')
    return signature


def summarize(root, cases=None, *, baseline_case='native_k24'):
    root = Path(root)
    cases = CASES if cases is None else {name:tuple(policy) for name,policy in cases.items()}
    baselines = {f'native_k{k}': (k, 0, 'full') for k in (0, 16, 20, 24)}
    baselines['full_k0'] = (0, 2, 'full')
    if (baseline_case not in baselines or not cases or next(iter(cases))!=baseline_case
            or cases[baseline_case]!=baselines[baseline_case]
            or any(not re.fullmatch(r'\w+',name) or len(policy)!=3 for name,policy in cases.items())):
        raise ValueError('Explicit named policies with native_k24 first required')
    statuses = {}
    for line in (root/'status.log').read_text().splitlines():
        match = re.fullmatch(r'(\w+) returncode=(\d+)', line)
        if not match or match[1] not in cases or match[1] in statuses:
            raise ValueError('Malformed/duplicate status')
        statuses[match[1]] = int(match[2])
    if set(statuses) != set(cases):
        raise ValueError('Screen incomplete')
    rows = []; signatures = set()
    for name, policy in cases.items():
        log = root/(name+'.log'); raw_log = log.read_bytes()
        row = dict(case=name, policy=dict(zip(('keep_full_layers', 'checkpoint_every', 'checkpoint_mode'), policy)),
            returncode=statuses[name], log_source=str(log), log_sha256=hashlib.sha256(raw_log).hexdigest())
        if statuses[name]:
            if not any(word in raw_log for word in (b'CUDA out of memory', b'torch.OutOfMemoryError', b'torch.cuda.OutOfMemoryError')):
                raise ValueError('Non-capacity failure is not a valid screen')
            rows.append(dict(row, status='oom')); continue
        path = root/name/'result.json'; raw = path.read_bytes(); result = json.loads(raw)
        signatures.add(audit_result(result, policy))
        rows.append(dict(row, status='passed', source=str(path), sha256=hashlib.sha256(raw).hexdigest(),
            median_step_seconds=result['median_step_seconds'], tokens_per_second=result['global_tokens_per_second'],
            peak_allocated_gib=max(x['peak_allocated_gib'] for x in result['records']),
            peak_reserved_gib=max(x['peak_reserved_gib'] for x in result['records']),
            initial_hf_parity=result['initial_hf_parity']))
    if len(signatures) != 1 or rows[0]['status'] != 'passed':
        raise ValueError('Matched passing baseline required')
    baseline = rows[0]['median_step_seconds']
    for row in rows:
        if row['status'] == 'passed':
            row['throughput_speedup_vs_native_k24'] = baseline/row['median_step_seconds']
            if baseline_case != 'native_k24':
                del row['throughput_speedup_vs_native_k24']
            row['throughput_speedup_vs_baseline'] = baseline/row['median_step_seconds']
    winner = min((r for r in rows if r['status'] == 'passed'), key=lambda r: r['median_step_seconds'])
    return dict(schema_version=1, complete=True, baseline_case=baseline_case,
        workload_signature=list(next(iter(signatures))),
        records=rows, screening_winner=winner['case'],
        scope='Fixed-workload single-round screen; not ABBA confirmation, convergence or universal optimum')


def main():
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case-spec',type=Path,help='Explicit refinement case map; keep budgets in separate rounds')
    p.add_argument('--baseline-case', choices=['native_k24', 'native_k0', 'native_k16', 'native_k20', 'full_k0'],
        default='native_k24')
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('Preserve previous report')
    result = summarize(a.root,json.loads(a.case_spec.read_text()) if a.case_spec else None,
        baseline_case=a.baseline_case)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
