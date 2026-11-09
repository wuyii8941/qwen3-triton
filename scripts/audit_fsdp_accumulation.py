"""Explicit effective-batch audit; never pool different training objectives."""
import json
import math
import re
import statistics


def audit_result(result, *, batch, accumulation, gradient_sync, policy):
    if (not result.get('complete') or result.get('backend')!='official'
            or result.get('batch')!=batch or result.get('accumulation')!=accumulation
            or result.get('gradient_sync')!=gradient_sync):
        raise ValueError('Complete worker with declared microbatch/sync policy required')
    if min(batch,accumulation)<1 or gradient_sync not in ('each','defer'):
        raise ValueError('Invalid accumulation dimensions')
    opt=result['optimization']
    if tuple(opt[k] for k in ('keep_full_layers','checkpoint_every','checkpoint_mode'))!=tuple(policy):
        raise ValueError('Mislabeled retention/checkpoint policy')
    if (opt.get('loss_chunk_size',0) or opt.get('ce_only_chunk_size',0)
            or opt.get('swiglu','native')!='native' or opt.get('exchange_reduce')):
        raise ValueError('Unrelated loss/operator change')
    world=result['world_size'];effective=world*batch*accumulation
    if result.get('effective_global_batch')!=effective:
        raise ValueError('Effective global batch mismatch')
    rows=result['records'];warm=result['warmup'];steps=result['steps']
    if warm<1 or steps<1 or len(rows)!=warm+steps:raise ValueError('Incomplete measurements')
    budget=result['memory_budget_gib']
    if not math.isfinite(budget) or budget<=0:raise ValueError('Explicit finite memory budget required')
    for i,row in enumerate(rows):
        if row['step']!=i+1 or row['warmup']!=(i<warm):raise ValueError('Step order mismatch')
        if not all(math.isfinite(row[k]) for k in
                ('max_rank_seconds','loss','peak_allocated_gib','peak_reserved_gib')):
            raise ValueError('Nonfinite measurement')
        if not (row['max_rank_seconds']>0 and
                0<=row['peak_allocated_gib']<=row['peak_reserved_gib']<=budget):
            raise ValueError('Invalid latency/memory')
    median=statistics.median(r['max_rank_seconds'] for r in rows if not r['warmup'])
    if not math.isclose(median,result['median_step_seconds'],rel_tol=1e-10):
        raise ValueError('Median mismatch')
    if not math.isclose(effective*result['seq']/median,result['global_tokens_per_second'],rel_tol=1e-10):
        raise ValueError('Effective-batch throughput mismatch')
    parity=result.get('initial_hf_parity',{}).get('max_abs',float('inf'))
    if not math.isfinite(parity) or not 0<=parity<=.03:raise ValueError('HF initial parity required')
    manifest=result['manifest']
    if not manifest.get('input_sha256') or not manifest.get('implementation_sources',{}).get('sha256'):
        raise ValueError('Input and implementation identity required')
    # Raw tensor bytes are identical across B2/A1 and B1/A2 when sample order
    # is held fixed; tensor shape is intentionally excluded from this digest.
    # Older workers only recorded rank0. Current workers also provide the full
    # rank digest vector; never mix the stronger and legacy evidence levels.
    rank_hashes=manifest.get('input_sha256_by_rank')
    if rank_hashes is not None and (len(rank_hashes)!=world or any(
            not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in rank_hashes)):
        raise ValueError('Complete all-rank input identity required')
    return tuple(result[k] for k in ('model','world_size','seq','warmup','steps','precision',
        'attention','memory_budget_gib'))+(effective,manifest['input_sha256'],
        manifest['implementation_sources']['sha256'],manifest['pytorch'],manifest['cuda'],
        manifest['gpu'],json.dumps(result.get('allocator',{}),sort_keys=True),tuple(policy),
        None if rank_hashes is None else tuple(rank_hashes),
        json.dumps(result.get('communication_environment',{}),sort_keys=True))


def compare(results, specs):
    if set(results)!=set(specs) or len(results)<2:raise ValueError('Complete explicit case map required')
    signatures={audit_result(results[name],**spec) for name,spec in specs.items()}
    if len(signatures)!=1:raise ValueError('Effective batch, data, source, runtime or policy mismatch')
    first=next(iter(specs));baseline=results[first]['median_step_seconds']
    return dict(complete=True,baseline=first,
        input_identity_scope='all_ranks' if all(r['manifest'].get('input_sha256_by_rank')
            is not None for r in results.values()) else 'rank0_legacy',
        throughput_ratios={n:baseline/r['median_step_seconds'] for n,r in results.items()},
        scope='Fixed effective-batch short-run comparison; not convergence or BF16 full-batch bitwise equivalence')
