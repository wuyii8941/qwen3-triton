"""Retry a recorded I/O-related timeout using the committed checkpoint only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.audit_fsdp_resume import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--predecessor',type=int,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**fields):
        (a.output/'status.json').write_text(json.dumps(dict(phase=phase,**fields),indent=2))
        print(phase,fields,flush=True)
    try:
        status('waiting_for_native_resume_exit')
        while True:
            try:cmd=Path(f'/proc/{a.predecessor}/cmdline').read_bytes()
            except FileNotFoundError:break
            if b'scripts.queue_real_fsdp_resume' not in cmd:break
            time.sleep(15)
        previous=json.loads((a.previous/'status.json').read_text())
        log=(a.previous/'replay.log').read_text()
        if (previous['phase']!='failed' or 'Watchdog caught collective operation timeout' not in log
                or 'Timeout(ms)=600000' not in log or 'replay mismatch' in log):
            raise ValueError('Only the recorded collective-timeout failure authorizes this retry')
        checkpoint=a.previous/'replay/step2'
        manifest=json.loads((checkpoint/'complete.json').read_text())
        if manifest['step']!=2 or manifest['world_size']!=4:raise ValueError('Committed step2 checkpoint required')
        status('waiting_for_idle_gpus')
        while True:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).splitlines()
            values=[tuple(int(x.strip()) for x in row.split(',')) for row in rows]
            if len(values)>=4 and all(m<2048 and u<10 for m,u in values[:4]):break
            time.sleep(15)
        status('running_existing_checkpoint_replay',reader='offset_sorted',wrote_checkpoint=False)
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (a.output/'replay.log').open('x') as stream:
            subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'benchmarks/verify_existing_fsdp_resume.py','--checkpoint',str(checkpoint),
                '--reader','offset_sorted','--output',str(a.output/'replay')],env=env,
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        report=audit(a.output/'replay')
        if not report['full_8b_resume_proven']:raise ValueError('Real 8B proof required')
        (a.output/'audit.json').write_text(json.dumps(report,indent=2))
        status('finished',reader='offset_sorted',result=str(a.output/'replay/result.json'),
            original_native_attempt='failed_collective_timeout',wrote_checkpoint=False)
    except Exception as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
