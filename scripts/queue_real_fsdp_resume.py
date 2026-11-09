"""Run real 8B checkpoint replay only after the named performance campaign."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predecessor',type=int,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**fields):
        (a.output/'status.json').write_text(json.dumps(dict(phase=phase,**fields),indent=2))
        print(phase,fields,flush=True)
    try:
        status('waiting_for_performance_campaign')
        while True:
            try:cmd=Path(f'/proc/{a.predecessor}/cmdline').read_bytes()
            except FileNotFoundError:break
            if b'scripts.confirm_fsdp_s512_candidates' not in cmd:break
            time.sleep(15)
        state=json.loads((a.campaign/'status.json').read_text())
        if state.get('phase')!='finished':raise ValueError('Predecessor did not finish successfully')
        status('waiting_for_idle_gpus')
        while True:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).splitlines()
            values=[tuple(int(x.strip()) for x in row.split(',')) for row in rows]
            if len(values)>=4 and all(m<2048 and u<10 for m,u in values[:4]):break
            time.sleep(15)
        target=a.output/'replay'
        probe=subprocess.run([sys.executable,'benchmarks/verify_real_fsdp_resume.py','--resource-plan',
            '--output',str(target)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
        plan=json.loads(probe.stdout)
        (a.output/'resource_plan.json').write_text(json.dumps(plan,indent=2))
        if not plan['resources_sufficient']:
            status('resource_guard_rejected',plan=plan);return
        status('running_real_8b_replay')
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (a.output/'replay.log').open('x') as stream:
            subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'benchmarks/verify_real_fsdp_resume.py','--output',str(target),
                '--keep-full-layers','32','--checkpoint-every','1'],env=env,
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        result=json.loads((target/'result.json').read_text())
        if not result.get('passed') or not result.get('full_8b_resume_proven'):
            raise ValueError('Real 8B replay did not establish required proof')
        status('finished',result=str(target/'result.json'),checkpoint_retained=True)
    except Exception as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
