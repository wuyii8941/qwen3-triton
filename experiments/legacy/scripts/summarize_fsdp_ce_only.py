"""Dedicated CE-only/retention screen audit; old native screens reject CE-only."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.summarize_fsdp_memory import audit_result


def summarize(root,cases):
    root=Path(root)
    if (not cases or next(iter(cases))!='baseline_full_k8' or tuple(cases['baseline_full_k8'])!=(8,2,'full',0)
            or any(not re.fullmatch(r'\w+',n) or len(p)!=4 or p[3] not in (0,128) for n,p in cases.items())):
        raise ValueError('Explicit CE-only case specification and native-loss full K8 baseline required')
    statuses={}
    for line in (root/'status.log').read_text().splitlines():
        match=re.fullmatch(r'(\w+) returncode=(\d+)',line)
        if not match or match[1] in statuses or match[1] not in cases:raise ValueError('Invalid status')
        statuses[match[1]]=int(match[2])
    if set(statuses)!=set(cases):raise ValueError('Incomplete screen')
    rows=[];signatures=set()
    for name,policy in cases.items():
        log=root/(name+'.log');raw_log=log.read_bytes()
        row=dict(case=name,policy=list(policy),returncode=statuses[name],log_source=str(log),
                 log_sha256=hashlib.sha256(raw_log).hexdigest())
        if statuses[name]:
            if not any(s in raw_log for s in (b'torch.OutOfMemoryError',b'CUDA out of memory')):
                raise ValueError('Non-capacity failure requires diagnosis')
            rows.append(dict(row,status='oom'));continue
        path=root/name/'result.json';raw=path.read_bytes();r=json.loads(raw)
        if r['optimization'].get('ce_only_chunk_size',0)!=policy[3]:raise ValueError('Mislabeled CE-only policy')
        signatures.add(audit_result(r,tuple(policy[:3]),allow_ce_only=True))
        rows.append(dict(row,status='passed',source=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            median_step_seconds=r['median_step_seconds'],tokens_per_second=r['global_tokens_per_second'],
            records=r['records'],initial_hf_parity=r['initial_hf_parity']))
    if len(signatures)!=1 or rows[0]['status']!='passed':raise ValueError('Matched passing baseline required')
    baseline=rows[0]['median_step_seconds']
    for row in rows:
        if row['status']=='passed':row['throughput_speedup_vs_baseline']=baseline/row['median_step_seconds']
    return dict(complete=True,records=rows,workload_signature=list(next(iter(signatures))),
        screening_winner=min((r for r in rows if r['status']=='passed'),key=lambda r:r['median_step_seconds'])['case'],
        scope='Single-round matched CE-only screen; not repeated confirmation, convergence or universal optimum')


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--case-spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=summarize(a.root,json.loads(a.case_spec.read_text()))
    with a.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
