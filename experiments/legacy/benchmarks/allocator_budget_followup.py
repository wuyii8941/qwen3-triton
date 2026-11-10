"""Paired native/consumed sweep with expandable segments, same allocator cap.

The initial default-allocator OOM had >1 GiB reserved/unallocated. Do not
declare a model's capacity limit without testing this normal software option.
"""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='artifacts/training_budget_expandable_wave2_v1');parser.add_argument('--full',action='store_true');a=parser.parse_args()
    root=Path(a.output);root.mkdir(exist_ok=False)
    records=[];env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    env.pop('CUBLAS_WORKSPACE_CONFIG',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    env['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    def run(variant,micro,name):
        output=root/f'{name}.json'
        with (root/f'{name}.log').open('w') as stream:
            result=subprocess.run([sys.executable,'-m','benchmarks.training_dataflow_wave2','step',
                '--variant',variant,'--effective','8','--micro',str(micro),'--budget','16',
                '--steps','12','--output',str(output)],env=env,stdout=stream,stderr=subprocess.STDOUT)
        row=json.loads(output.read_text()) if output.exists() else dict(status='error',error='No worker result')
        row.update(name=name,returncode=result.returncode)
        records.append(row)
        (root/'result.json').write_text(json.dumps(dict(complete=False,records=records),indent=2))
        print('EXPANDABLE',name,row.get('status'),row.get('tokens_per_second'),flush=True)
        if row.get('status')=='error':raise RuntimeError(f'Non-OOM failure: inspect {name} log')
        return row
    for micro in ((1,2,4,8) if a.full else (4,8)):
        for variant in ('baseline','consumed'):run(variant,micro,f'{variant}_mb{micro}')
    selected={}
    for variant in ('baseline','consumed'):
        good=[r for r in records if r.get('variant')==variant and r.get('status')=='ok']
        if not good:
            row=run(variant,2,f'{variant}_mb2')
            if row.get('status')=='ok':good=[row]
        if good:selected[variant]=max(good,key=lambda r:r['tokens_per_second'])['microbatch']
    for variant in ('consumed','baseline'):
        if variant in selected:run(variant,selected[variant],f'confirm_{variant}_mb{selected[variant]}')
    (root/'result.json').write_text(json.dumps(dict(complete=True,records=records,selected=selected),indent=2))


if __name__=='__main__':main()
