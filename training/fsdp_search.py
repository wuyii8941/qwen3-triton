"""CPU-only validation/selection for memory-bounded FSDP experiments.

Selection is scoped to an identical workload, not a claim of global optimality.
OOM is a capacity result; numerical/communication errors must not be hidden.
"""
import math
import json


def retention_candidates(layer_count):
    if layer_count<1:raise ValueError('Positive decoder layer count required')
    return sorted({k for k in (0,4,8,12,16,24,layer_count) if k<=layer_count})


def workload_signature(result):
    return tuple(result[k] for k in ('model','world_size','batch','seq','steps','warmup',
        'precision','attention','memory_budget_gib'))+(result['manifest']['input_sha256'],
            result['manifest'].get('implementation_sources',{}).get('sha256'),
            json.dumps(result.get('allocator',{}),sort_keys=True),
            result.get('accumulation',1),result.get('gradient_sync','each'),
            result.get('optimization',{}).get('checkpoint_every',0),
            result.get('optimization',{}).get('checkpoint_mode','full'),
            result.get('optimization',{}).get('ce_only_chunk_size',0),
            result.get('optimization',{}).get('loss_chunk_size',0))


def select_retention(records,budget):
    if not records or not math.isfinite(budget) or budget<=0:raise ValueError('Invalid search')
    signatures=set();valid=[];seen=set()
    for row in records:
        k=row['keep_full_layers']
        if k in seen or k<0:raise ValueError('Duplicate/invalid candidate')
        seen.add(k)
        if row['status']=='oom':continue
        if row['status']!='passed':raise ValueError('Unresolved non-capacity failure')
        r=row['result']
        if not r.get('complete') or r['optimization']['keep_full_layers']!=k:
            raise ValueError('Incomplete/mislabeled worker result')
        signatures.add(workload_signature(r))
        if r['memory_budget_gib']!=budget:raise ValueError('Budget mismatch')
        latency=r['median_step_seconds']
        peak=max(x['peak_allocated_gib'] for x in r['records'])
        parity=r.get('initial_hf_parity',{}).get('max_abs',float('inf'))
        if not (math.isfinite(latency) and latency>0 and math.isfinite(peak) and 0<=peak<=budget
                and math.isfinite(parity) and 0<=parity<=.03):
            raise ValueError('Invalid latency/memory/parity')
        valid.append(row)
    if len(signatures)!=1 or not any(r['keep_full_layers']==0 for r in valid):
        raise ValueError('Matched passing K=0 baseline required')
    best=min(valid,key=lambda x:x['result']['median_step_seconds'])
    base=next(r for r in valid if r['keep_full_layers']==0)
    return dict(best_keep_full_layers=best['keep_full_layers'],
        median_step_seconds=best['result']['median_step_seconds'],
        baseline_seconds=base['result']['median_step_seconds'],
        throughput_speedup=base['result']['median_step_seconds']/best['result']['median_step_seconds'],
        scope='Screening winner only; requires interleaved repeated confirmation and convergence validation')
