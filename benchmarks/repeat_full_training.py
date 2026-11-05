"""Randomized isolated-process rounds; preserve every trace and result.

Each process reloads identical original weights. This is cross-process paired
rounds, not an in-process per-step crossover; report uncertainty accordingly.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tag',required=True)
    p.add_argument('--seq',type=int,default=512)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--variants',default='hf,compile,rms,swiglu,qk,operators,fused_loss')
    p.add_argument('--adamw-math',choices=['legacy','rounded'],default='legacy')
    a=p.parse_args()
    target=Path('artifacts')/f'full_training_repeated_{a.tag}.json'
    if target.exists():raise FileExistsError(target)
    rng=random.Random(51)
    report=dict(seq=a.seq,adamw_math=a.adamw_math,
        scope='Randomized process-isolated rounds, 8 steady steps each, original weights reloaded',runs=[])
    for repeat in range(a.rounds):
        order=a.variants.split(',');rng.shuffle(order)
        for variant in order:
            stem=f'full_training_repeated_{a.tag}_r{repeat}_{variant}'
            output=Path('artifacts')/f'{stem}.json';log=output.with_suffix('.log')
            if output.exists() or log.exists():raise FileExistsError(stem)
            base_variant=variant[:-6] if variant.endswith('_adamw') else variant
            cmd=[sys.executable,'benchmarks/benchmark_full_training.py','--variant',base_variant,
                 '--seq',str(a.seq),'--timing','steady','--steps','8','--output',str(output)]
            if variant.endswith('_adamw'):cmd+=['--optimizer','triton','--adamw-math',a.adamw_math]
            print('start',repeat,variant,flush=True)
            with log.open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            report['runs'].append(dict(round=repeat,variant=variant,returncode=r.returncode,
                artifact=str(output),log=str(log)))
            target.write_text(json.dumps(report,indent=2))
            if r.returncode:raise RuntimeError(log)
            print('done',repeat,variant,flush=True)


if __name__=='__main__':main()
