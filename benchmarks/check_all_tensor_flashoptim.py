"""Persistent multi-tensor launch ablation; fresh process per stage."""
import json
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--vectorized',action='store_true');parser.add_argument('--async-metadata',action='store_true');args=parser.parse_args()
    prefix='all_vec_async' if args.async_metadata else ('all_vec' if args.vectorized else 'all')
    tag='all_tensor_flashoptim_async_v1' if args.async_metadata else ('all_tensor_flashoptim_vectorized_v1' if args.vectorized else 'all_tensor_flashoptim_v1')
    root=Path('artifacts')/tag;root.mkdir(exist_ok=False)
    rows=[]
    for i,variant in enumerate(('baseline',prefix,prefix+'_both',prefix+'_both',prefix,'baseline')):
        output=root/f'{i}_{variant}.json'
        with (root/f'{i}_{variant}.log').open('w') as stream:
            command=[sys.executable,'-m','benchmarks.benchmark_tied_embedding_step','--single-stage',variant,
                     '--output',str(output),'--steps','20']
            if i in (0,1):command+=['--profile']
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            (root/'error.json').write_text(json.dumps(dict(stage=variant,returncode=result.returncode)))
            raise RuntimeError('Stage failed; inspect log')
        row=json.loads(output.read_text());row['order']=i;rows.append(row)
        (root/'result.json').write_text(json.dumps(dict(complete=False,rows=rows),indent=2))
        print('ALL_TENSOR',i,variant,row['median_step_ms'],flush=True)
    baseline=[r for r in rows if r['stage']=='baseline']
    comparisons=[]
    for variant in (prefix,prefix+'_both'):
        candidate=[r for r in rows if r['stage']==variant]
        comparisons.append(dict(variant=variant,pairs=[dict(
            baseline_ms=b['median_step_ms'],candidate_ms=c['median_step_ms'],
            time_reduction=1-c['median_step_ms']/b['median_step_ms'],
            peak_saved_mib=(b['peak_allocated_bytes']-c['peak_allocated_bytes'])/2**20,
            training_loss_max_abs=max(abs(x-y) for x,y in zip(b['losses'],c['losses'])))
            for b,c in zip(baseline,candidate)]))
    (root/'result.json').write_text(json.dumps(dict(complete=True,rows=rows,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
