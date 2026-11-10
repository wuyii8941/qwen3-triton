"""Serial candidate launcher; preserve baseline and wait for idle GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wait-pid',type=int,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--screen-prefetch',action='store_true',help='Gate experimental prefetch in the idle gap before candidate')
    args=p.parse_args()
    status_path=args.output.parent/(args.output.name+'_queue.json')
    def status(phase,**extra):
        status_path.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),updated=time.time(),**extra),indent=2))
        print(phase,flush=True)
    try:
        status('waiting_for_baseline')
        proc=Path(f'/proc/{args.wait_pid}/cmdline')
        while proc.exists():
            try:command=proc.read_bytes()
            except FileNotFoundError:break
            if b'fsdp_convergence_job.py' not in command:break
            time.sleep(15)
        baseline=json.loads(args.baseline.read_text())
        if not baseline.get('complete') or not baseline.get('best_parameters_restored'):
            raise RuntimeError('Baseline did not finish correctly; candidate not started')
        if args.output.exists():raise FileExistsError('Candidate output already exists')
        status('waiting_for_idle_gpus')
        while True:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True)
            usage=[tuple(int(v.strip()) for v in row.split(',')) for row in text.splitlines()]
            if len(usage)>=4 and all(memory<2048 and utilization<10 for memory,utilization in usage[:4]):break
            time.sleep(15)
        if args.screen_prefetch:
            status('prefetch_gpu_gate')
            root=args.output.parent/'prefetch_screen'
            root.mkdir(exist_ok=False)
            gate=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'benchmarks/verify_block_fsdp.py','--bf16','--prefetch','--checkpoint-test','--export-test']
            with (root/'gate.log').open('w') as stream:
                checked=subprocess.run(gate,stdout=stream,stderr=subprocess.STDOUT)
            # A failed experiment must not replace the validated training path.
            report=dict(gpu_gate_passed=checked.returncode==0)
            if checked.returncode==0:
                status('prefetch_performance_screen')
                with (root/'screen.log').open('w') as stream:
                    screen=subprocess.run([sys.executable,'scripts/run_fsdp_comparison.py',
                        '--cases','official','custom_prefetch','--steps','6','--repeats','1',
                        '--profile-first','--reference-check','--output',str(root/'matrix')],stdout=stream,stderr=subprocess.STDOUT)
                report['screen_returncode']=screen.returncode
            (root/'status.json').write_text(json.dumps(report,indent=2))
        status('running')
        with (args.output.parent/(args.output.name+'.log')).open('w') as stream:
            subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'training/fsdp_convergence_job.py','--backend','custom_avg','--output',str(args.output)],
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        result=json.loads((args.output/'result.json').read_text())
        status('finished',complete=result.get('complete'),converged=result.get('converged'),
            stop_reason=result.get('stop_reason'))
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
