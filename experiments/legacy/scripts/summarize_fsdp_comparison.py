"""Summarize complete matched rounds; do not pool different input workloads."""
import argparse
import json
import math
from pathlib import Path


def summarize(matrix):
    if not matrix.get('complete'):raise ValueError('Comparison round incomplete')
    rows=matrix['records']
    if {r['backend'] for r in rows}!={'custom','custom_avg','official','official_checkpoint'}:
        raise ValueError('Missing required comparison backend')
    signatures=set();times={};memory={}
    for record in rows:
        r=record['result']
        if record['returncode'] or not r.get('complete'):raise ValueError('Invalid worker')
        signatures.add((r['model'],r['seq'],r['batch'],r['world_size'],r['steps'],r['warmup'],
            r['precision'],r['attention'],r['manifest']['input_sha256'],
            r['manifest'].get('implementation_sources',{}).get('sha256'),
            r.get('accumulation',1),r.get('gradient_sync','each'),
            r.get('optimization',{}).get('loss_chunk_size',0)))
        value=r['median_step_seconds']
        if not math.isfinite(value) or value<=0:raise ValueError('Invalid latency')
        times[record['backend']]=value
        memory[record['backend']]=max(x['peak_allocated_gib'] for x in r['records'])
    if len(signatures)!=1:raise ValueError('Unmatched workloads')
    best=min(times['official'],times['official_checkpoint'])
    return dict(seconds=times,peak_allocated_gib=memory,
        custom_speedup_vs_best_official=best/times['custom'],
        avg_speedup_vs_best_official=best/times['custom_avg'],
        avg_speedup_vs_custom=times['custom']/times['custom_avg'],
        scope='Steady-state only; not convergence and not a statistical significance test')


def main():
    p=argparse.ArgumentParser();p.add_argument('rounds',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=dict(rounds=[dict(source=str(path),summary=summarize(json.loads(path.read_text())))
        for path in args.rounds])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
