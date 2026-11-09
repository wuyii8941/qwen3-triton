"""Serial large-model comparison; isolated processes, small logs/results only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
    p.add_argument('--cases',nargs='+',default=['official','custom','official_checkpoint','custom_avg'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--profile-first',action='store_true')
    p.add_argument('--batch',type=int,default=1,help='Same microbatch per rank for every backend')
    p.add_argument('--seq',type=int,default=512)
    p.add_argument('--reference-check',action='store_true')
    args=p.parse_args()
    if min(args.batch,args.seq,args.steps,args.repeats)<1:p.error('Positive workload values required')
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for repeat in range(args.repeats):
        order=args.cases if repeat%2==0 else list(reversed(args.cases))
        for backend in order:
            # Never stop unrelated jobs or launch over an occupied large model.
            while True:
                usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True)
                readings=[tuple(int(v.strip()) for v in line.split(',')) for line in usage.splitlines()]
                if len(readings)>=4 and all(memory<2048 and utilization<10 for memory,utilization in readings[:4]):break
                time.sleep(15)
            output=args.output/f'r{repeat}_{backend}'
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'benchmarks/compare_fsdp_training.py','--backend',backend,'--model',args.model,
                '--seq',str(args.seq),'--batch',str(args.batch),'--warmup','3',
                '--steps',str(args.steps),'--output',str(output)]
            if args.profile_first and repeat==0:command.append('--profile')
            # Experimental prefetch must pass real-model parity before timing;
            # apply the same untimed check to its paired official baseline.
            if args.reference_check or 'custom_prefetch' in args.cases:
                command.append('--reference-check')
            with (args.output/f'r{repeat}_{backend}.log').open('w') as stream:
                job=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            row=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else {}
            records.append(dict(repeat=repeat,backend=backend,returncode=job.returncode,result=row))
            (args.output/'matrix.json').write_text(json.dumps(dict(complete=False,records=records),indent=2))
            print(json.dumps(dict(backend=backend,repeat=repeat,returncode=job.returncode,
                seconds=row.get('median_step_seconds'))),flush=True)
            if job.returncode or not row.get('complete'):raise RuntimeError(f'Invalid case: {backend}')
    # Manifest digest is per-rank0 data and must match across identical work.
    hashes={r['result']['manifest']['input_sha256'] for r in records}
    if len(hashes)!=1:raise RuntimeError('Input mismatch across backends')
    (args.output/'matrix.json').write_text(json.dumps(dict(complete=True,records=records),indent=2))


if __name__=='__main__':main()
