"""Re-read 40GiB study records; partial audits never issue speedup decisions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.screen_fsdp_exchange_40 import CASES,decisions
from scripts.confirm_fsdp_exchange import audit_case
from scripts.audit_fsdp_exchange_confirmation import callback_contract
from scripts.fsdp_resource_budget import budget_ready


def audit_resources(path):
    raw=path.read_bytes();record=json.loads(raw)
    if any(record.get(k)!=v for k,v in dict(allocator_budget_gib=40.,free_margin_gib=2.5,idle_confirmations=3).items()):
        raise ValueError('Explicit 40GiB launch policy required')
    if len(record['samples'])!=3:raise ValueError('Three launch observations required')
    previous=0.
    for row in record['samples']:
        tick=row['unix_seconds'];readings=row['readings']
        if not math.isfinite(tick) or tick<=previous or len(readings)!=4 or any(
                len(r)!=3 or any(type(v) is not int or v<0 for v in r) for r in readings):
            raise ValueError('Invalid resource observations')
        if not budget_ready(readings,allocator_gib=40.,free_margin_gib=2.5):
            raise ValueError('Launch memory/utilization guard did not pass')
        previous=tick
    return hashlib.sha256(raw).hexdigest()


def audit(root,*,allow_partial=False):
    path=root/'result.json';complete=path.exists()
    summary=json.loads(path.read_text()) if complete else None
    if not complete and not allow_partial:raise ValueError('Study incomplete; use explicit diagnostic partial mode')
    if complete and (summary.get('complete') is not True or summary['memory_budget_gib']!=40 or
            summary['case_specs']!={n:list(s) for n,s in CASES.items()}):
        raise ValueError('Wrong completed study contract')
    rows=summary['records'] if complete else json.loads((root/'progress.json').read_text())
    expected=[(r,n) for r,order in ((0,list(CASES)),(1,list(reversed(CASES)))) for n in order]
    if [(r['repeat'],r['case']) for r in rows]!=(expected if complete else expected[:len(rows)]):
        raise ValueError('Wrong ordered study/prefix')
    signatures=set();runs={};case_records=[];resources={}
    for row in rows:
        repeat,name=row['repeat'],row['case'];label=f'r{repeat}_{name}'
        resources[label]=audit_resources(root/(label+'_resources.json'))
        log=(root/(label+'.log')).read_bytes()
        if hashlib.sha256(log).hexdigest()!=row['log_sha256']:raise ValueError('Changed log')
        if row['status']=='oom':
            if row['returncode']==0 or not any(t in log for t in
                    (b'CUDA out of memory',b'torch.OutOfMemoryError',b'torch.cuda.OutOfMemoryError')):
                raise ValueError('Invalid capacity failure')
            case_records.append(dict(case=label,status='oom'));continue
        result_path=root/label/'result.json';raw=result_path.read_bytes();result=json.loads(raw)
        if (row['status']!='passed' or row['returncode']!=0 or
                Path(row['source']).resolve()!=result_path.resolve() or hashlib.sha256(raw).hexdigest()!=row['sha256']):
            raise ValueError('Result provenance mismatch')
        if result['memory_budget_gib']!=40:raise ValueError('Do not relabel other budgets')
        signatures.add(audit_case(result,CASES[name]));callback_contract(result,CASES[name])
        manifest=result['manifest'];snapshot=manifest['implementation_sources']
        encoded=json.dumps(snapshot['files'],sort_keys=True,separators=(',',':')).encode()
        if (hashlib.sha256(encoded).hexdigest()!=snapshot['sha256'] or
                manifest['source_sha256']!=snapshot['files']['benchmarks/compare_fsdp_training.py'] or
                manifest['input_sha256_by_rank'][0]!=manifest['input_sha256']):
            raise ValueError('Source/input identity inconsistent')
        if complete:
            if snapshot!=summary['implementation_sources']:raise ValueError('Supervisor/worker source mismatch')
            for filename,digest in manifest['exchange_sources'].items():
                if summary['helper_sources'].get(filename)!=digest:raise ValueError('Helper source mismatch')
        if (row['median_step_seconds']!=result['median_step_seconds'] or
                row['peak_allocated_gib']!=max(r['peak_allocated_gib'] for r in result['records']) or
                row['peak_reserved_gib']!=max(r['peak_reserved_gib'] for r in result['records'])):
            raise ValueError('Summary metrics differ from raw records')
        measured=[r for r in result['records'] if not r['warmup']]
        def counter(key):
            values=[r.get('allocator_activity_max_rank',{}).get(key) for r in measured]
            return max(values) if all(v is not None for v in values) else None
        case_records.append(dict(case=label,status='passed',median_step_seconds=result['median_step_seconds'],
            measured_min_seconds=min(r['max_rank_seconds'] for r in measured),
            measured_max_seconds=max(r['max_rank_seconds'] for r in measured),
            peak_allocated_gib=row['peak_allocated_gib'],peak_reserved_gib=row['peak_reserved_gib'],
            max_allocator_retries_measured=counter('num_alloc_retries'),
            max_allocator_syncs_measured=counter('num_sync_all_streams'),
            initial_hf_logits_max_abs=result['initial_hf_parity']['max_abs']))
        runs[(repeat,name)]=result
    if not signatures or len(signatures)!=1:raise ValueError('Matched passing raw records required')
    calculated={};protocol_ratios=[]
    if complete:
        if any((r,'native_auto_k32') not in runs for r in (0,1)):
            raise ValueError('Both passing native auto32 controls required')
        calculated=decisions(rows)
        if calculated!=summary['decisions']:raise ValueError('Decision differs from raw metrics')
        for repeat in (0,1):
            by_proto={proto:[v['median_step_seconds'] for (r,n),v in runs.items()
                if r==repeat and not CASES[n][1] and CASES[n][3]==proto] for proto in (None,'Simple')}
            protocol_ratios.append(min(by_proto[None])/min(by_proto['Simple']) if by_proto['Simple'] else None)
    return dict(complete=complete,partial_diagnostic_only=not complete,memory_budget_gib=40,
        raw_records_audited=len(rows),records=case_records,decisions=calculated,
        best_tested_native_simple_over_auto_ratios=protocol_ratios,
        launch_resource_sha256=resources,defaults_changed=False,convergence_proven=False,
        original_42gib_study_replaced=False,
        scope='Raw record/launch/allocator audit for this40GiB study only. Partial mode has no speedup decisions; no causal allocator attribution, external-load isolation, convergence or42GiB claim')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();result=audit(args.root,allow_partial=args.allow_partial)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
