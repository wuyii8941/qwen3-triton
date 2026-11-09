"""Audit A/B/C/C/B/A confirmation and separate untimed trace evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.summarize_fsdp_memory import audit_result


POLICIES = {'native_k24': (24,0,'full'), 'native_k30': (30,0,'full'), 'full_k35': (35,2,'full')}
ORDER = ('r0_native_k24','r0_native_k30','r0_full_k35',
         'r1_full_k35','r1_native_k30','r1_native_k24')


def merge(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if not all(math.isfinite(x) for x in (start,end)) or end<start:
            raise ValueError('Invalid trace interval')
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return merged


def profile_summary(trace):
    kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and 'dur' in e]
    comm=[e for e in kernels if 'nccl' in e.get('name','').lower()]
    local=[e for e in kernels if 'nccl' not in e.get('name','').lower()]
    def intervals(events):return merge([(e['ts'],e['ts']+e['dur']) for e in events])
    c,l=intervals(comm),intervals(local)
    i=j=0;overlap=0.
    while i<len(c) and j<len(l):
        overlap+=max(0.,min(c[i][1],l[j][1])-max(c[i][0],l[j][0]))
        if c[i][1]<l[j][1]:i+=1
        else:j+=1
    comm_union=sum(b-a for a,b in c);local_union=sum(b-a for a,b in l)
    runtime=[e.get('name','').lower() for e in trace['traceEvents']
             if e.get('cat') in ('cuda_runtime','cuda_driver')]
    allocation=[e for e in trace['traceEvents'] if e.get('cat') in ('cuda_runtime','cuda_driver')
                and e.get('name','').lower().startswith(('cudamalloc','cudafree'))]
    scopes=[e for e in trace['traceEvents'] if e.get('cat')=='user_annotation'
            and e.get('name') in ('comparison.forward','comparison.backward','comparison.optimizer') and 'dur' in e]
    scoped=[e for e in allocation if any(s['ts']<=e['ts']<s['ts']+s['dur'] for s in scopes)]
    names=[e.get('name','').lower().replace('_','') for e in comm]
    return dict(kernel_count=len(kernels),all_gather_kernel_count=sum('allgather' in n for n in names),
        reduce_scatter_kernel_count=sum('reducescatter' in n for n in names),
        summed_kernel_work_ms=sum(e['dur'] for e in kernels)/1000,
        comm_union_ms=comm_union/1000,local_union_ms=local_union/1000,
        comm_local_overlap_ms=overlap/1000,gpu_kernel_union_ms=(comm_union+local_union-overlap)/1000,
        cuda_malloc_calls=sum(n.startswith('cudamalloc') for n in runtime),
        cuda_free_calls=sum(n.startswith('cudafree') for n in runtime),
        allocation_api_summed_duration_ms=sum(e.get('dur',0) for e in allocation)/1000,
        training_scoped_allocation_calls=len(scoped) if scopes else None,
        training_scoped_allocation_summed_duration_ms=sum(e.get('dur',0) for e in scoped)/1000 if scopes else None,
        virtual_memory_driver_calls={name:sum(n.startswith(name.lower()) for n in runtime)
            for name in ('cuMemCreate','cuMemMap','cuMemUnmap','cuMemRelease')},
        scope='One extra rank0 untimed profile; sums are not critical-path savings; allocation API duration may include GPU waits/profiler overhead')


def summarize(root):
    root=Path(root)
    expected=[name+' returncode=0' for name in ORDER]
    if (root/'status.log').read_text().splitlines()!=expected:
        raise ValueError('Complete successful prescribed confirmation order required')
    rows=[];signatures=set();indexed={}
    for name in ORDER:
        policy=name[3:];path=root/name/'result.json';raw=path.read_bytes();r=json.loads(raw)
        if tuple(r[k] for k in ('world_size','batch','seq','warmup','steps','memory_budget_gib'))!=(4,1,512,3,12,42):
            raise ValueError('Confirmation workload mismatch')
        if (r['precision']!='FP32 master/AdamW/reduce, BF16 compute'
                or r['attention']!='forced Flash SDPA'):
            raise ValueError('Confirmation numerical protocol mismatch')
        signatures.add(audit_result(r,POLICIES[policy]))
        log=root/(name+'.log')
        row=dict(case=name,policy=policy,source=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            median_step_seconds=r['median_step_seconds'],tokens_per_second=r['global_tokens_per_second'],
            steady_step_seconds=[x['max_rank_seconds'] for x in r['records'] if not x['warmup']],
            peak_allocated_gib=max(x['peak_allocated_gib'] for x in r['records']),
            peak_reserved_gib=max(x['peak_reserved_gib'] for x in r['records']),
            initial_hf_parity=r['initial_hf_parity'])
        if name.startswith('r0_'):
            trace_path=root/name/'trace.json';trace=trace_path.read_bytes()
            row['profile']=dict(source=str(trace_path),sha256=hashlib.sha256(trace).hexdigest(),
                                **profile_summary(json.loads(trace)))
        rows.append(row);indexed[name]=row
    if len(signatures)!=1:raise ValueError('Inputs, implementation, environment or workload changed')
    pairs=[]
    for repeat in (0,1):
        seconds={p:indexed[f'r{repeat}_{p}']['median_step_seconds'] for p in POLICIES}
        pairs.append(dict(repeat=repeat,
            throughput_speedup_vs_native_k24=seconds['native_k24']/seconds['full_k35'],
            throughput_speedup_vs_native_k30=seconds['native_k30']/seconds['full_k35']))
    return dict(schema_version=1,complete=True,workload_signature=list(next(iter(signatures))),
        records=rows,paired_comparisons=pairs,
        candidate_over_two_percent_in_both_orders=all(p[k]>1.02 for p in pairs for k in
            ('throughput_speedup_vs_native_k24','throughput_speedup_vs_native_k30')),
        candidate_convergence_proven=False,
        scope='Two-order fixed-workload confirmation; not statistical significance or convergence')


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous report')
    r=summarize(a.root);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
