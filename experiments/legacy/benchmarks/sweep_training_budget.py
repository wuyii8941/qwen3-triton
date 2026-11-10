"""Isolated-process allocator-budget sweep with fixed effective batch size.

This is an explicitly imposed PyTorch allocator cap, not a claim about total
device usage including CUDA contexts/libraries. All variants share the cap.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tag',required=True)
    p.add_argument('--budget',type=float,default=28.)
    p.add_argument('--seq',type=int,default=2048)
    p.add_argument('--steps',type=int,default=8)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    target=root/'artifacts'/f'training_budget_{a.tag}.json'
    if target.exists():raise FileExistsError(target)
    cases=[(v,b) for v in ('hf','compile','fused_loss') for b in (1,2,4)]
    random.Random(41).shuffle(cases)
    report=dict(scope='Fixed effective batch4, seq2048 by default; no checkpointing; BF16 Adam states',
        budget_gib=a.budget,effective_batch=4,seq=a.seq,order=cases,cases=[])
    target.write_text(json.dumps(report,indent=2))
    for variant,batch in cases:
        stem=f'training_budget_{a.tag}_{variant}_b{batch}'
        out=root/'artifacts'/f'{stem}.json'
        log=root/'artifacts'/f'{stem}.log'
        if out.exists() or log.exists():raise FileExistsError(stem)
        cmd=[sys.executable,'benchmarks/benchmark_full_training.py','--variant',variant,
            '--batch',str(batch),'--seq',str(a.seq),'--accumulation',str(4//batch),
            '--memory-budget-gib',str(a.budget),'--timing','steady','--steps',str(a.steps),'--output',str(out)]
        print('start',variant,batch,flush=True)
        with log.open('x') as f:result=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
        text=log.read_text()
        status='ok' if result.returncode==0 else ('oom' if 'out of memory' in text.lower() else 'error')
        row=dict(variant=variant,microbatch=batch,accumulation=4//batch,status=status,
            returncode=result.returncode,command=cmd,log=str(log),artifact=str(out))
        if status=='ok':row['medians']=json.loads(out.read_text())['medians']
        report['cases'].append(row)
        target.write_text(json.dumps(report,indent=2))
        print('done',variant,batch,status,flush=True)
        if status=='error':raise RuntimeError(f'Non-OOM failure; inspect {log}')
    report['best_tested']={v:max((r['microbatch'] for r in report['cases'] if r['variant']==v and r['status']=='ok'),default=0)
        for v in ('hf','compile','fused_loss')}
    target.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
