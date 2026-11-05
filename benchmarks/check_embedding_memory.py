"""Locate the S2048 memory regression and check proxy lifetime, isolated workers."""
import json
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seq',type=int,default=2048);parser.add_argument('--confirm',action='store_true');args=parser.parse_args()
    tag='embedding_memory_lifetime_v1' if args.seq==2048 else f'embedding_memory_lifetime_s{args.seq}_v1'
    root=Path('artifacts')/(tag.replace('_v1','_confirm_v1') if args.confirm else tag);root.mkdir(exist_ok=False)
    rows=[]
    for variant in (('consume','candidate','baseline') if args.confirm else ('baseline','candidate','consume')):
        gate=f'artifacts/tied_embedding_consume_s{args.seq}_gate_v1/result.json' if variant=='consume' else (
            'artifacts/tied_embedding_deterministic_v1/result.json' if args.seq==512 else 'artifacts/tied_embedding_gate_b1_s2048_v1/result.json')
        if not json.loads(Path(gate).read_text())['passed']: raise RuntimeError('Gradient gate failed')
        output=root/f'{variant}.json'
        with (root/f'{variant}.log').open('w') as stream:
            result=subprocess.run([sys.executable,'-m','benchmarks.benchmark_tied_embedding_step',
                '--single-stage',variant,'--seq',str(args.seq),'--gate',gate,'--output',str(output),
                '--phase-memory'],stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{variant} failed: inspect log')
        row=json.loads(output.read_text());rows.append(row)
        (root/'result.json').write_text(json.dumps(dict(complete=False,rows=rows),indent=2))
        print('MEMORY',variant,row['phase_memory_bytes'],flush=True)
    (root/'result.json').write_text(json.dumps(dict(complete=True,rows=rows),indent=2))


if __name__=='__main__':main()
