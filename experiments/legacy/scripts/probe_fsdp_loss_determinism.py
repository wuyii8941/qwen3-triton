"""Recheck prior loss failures under deterministic numerics, after timing jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def classify(code, text):
    if code == 0:
        rows=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
        if len(rows)!=2 or {r.get('tied') for r in rows}!={False,True}:
            raise ValueError('Missing complete tied/untied success evidence')
        for row in rows:
            if not (row.get('passed') is True and row.get('deterministic') is True
                    and row.get('loss_chunk_size')==7 and row.get('seq')==11
                    and row.get('head_dim')==128 and row.get('steps')==3):
                raise ValueError('Unexpected success evidence')
        return 'passed'
    if 'AssertionError: Parameter ' in text and 'Tensor-likes are not close!' in text:
        return 'gradient_parity_failed'
    raise ValueError('Unexpected failure; not a numerical negative result')


def main():
    p=argparse.ArgumentParser();p.add_argument('--predecessor',type=int,required=True)
    p.add_argument('--confirmation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**extra):
        (args.output/'status.json').write_text(json.dumps(dict(phase=phase,updated=time.time(),**extra),indent=2))
        print(phase,flush=True)
    try:
        status('waiting_for_confirmation',predecessor=args.predecessor)
        path=Path(f'/proc/{args.predecessor}/cmdline')
        while path.exists():
            try:command=path.read_bytes()
            except FileNotFoundError:break
            if b'scripts.confirm_fsdp_shape' not in command:break
            time.sleep(15)
        previous=json.loads((args.confirmation/'status.json').read_text())
        if previous['phase']!='finished':raise ValueError('Timing confirmation did not finish successfully')
        env=dict(os.environ);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
        rows=[]
        for impl in ('default','native_ce'):
            status('waiting_for_idle_gpus',next_case=impl)
            while True:
                readings=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True).splitlines()
                values=[tuple(int(x.strip()) for x in line.split(',')) for line in readings]
                if len(values)>=4 and all(m<2048 and u<10 for m,u in values[:4]):break
                time.sleep(15)
            status('running',case=impl)
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'benchmarks/verify_block_fsdp.py','--backend','official','--bf16','--deterministic',
                '--head-dim','128','--seq','11','--keep-full-layers','1','--loss-chunk-size','7',
                '--loss-backward-impl',impl]
            log=args.output/f'{impl}.log'
            with log.open('x') as stream:
                code=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            rows.append(dict(implementation=impl,returncode=code,status=classify(code,log.read_text()),log=str(log)))
        (args.output/'result.json').write_text(json.dumps(dict(complete=True,records=rows,
            scope='Deterministic small-model gradient diagnostic; no performance or real-model quality claim'),indent=2)+'\n')
        status('finished')
    except Exception as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
