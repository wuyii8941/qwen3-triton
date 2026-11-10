"""Serial, memory-bounded retention sweep, then ABBA winner confirmation.

Waits for an identified predecessor and idle GPUs; never stops other jobs.
Only expected capacity OOMs are skipped. Other failures stop the search.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from training.fsdp_search import retention_candidates,select_retention


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,default=Path('/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-pid',type=int)
    p.add_argument('--wait-result',type=Path)
    p.add_argument('--budget-gib',type=float,default=42.)
    p.add_argument('--seq',type=int,default=512)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--screen-steps',type=int,default=6)
    p.add_argument('--confirm-steps',type=int,default=12)
    p.add_argument('--checkpoint-gate',action='store_true',help='Run tiny CUDA FSDP DCP replay before screening')
    a=p.parse_args()
    if a.wait_pid and not a.wait_result:p.error('--wait-result required with --wait-pid')
    if min(a.seq,a.batch,a.screen_steps,a.confirm_steps,a.budget_gib)<=0:p.error('Positive workload/budget required')
    layers=json.loads((a.model/'config.json').read_text())['num_hidden_layers']
    candidates=retention_candidates(layers)
    a.output.mkdir(parents=True,exist_ok=False)
    state=dict(complete=False,pid=os.getpid(),model=str(a.model),seq=a.seq,batch=a.batch,
        budget_gib=a.budget_gib,candidates=candidates,screen=[],confirmation=[],
        scope='Empirical retention optimum among tested configurations; not global optimality')
    def save(phase,**extra):
        state.update(phase=phase,updated=time.time(),**extra)
        target=a.output/'status.json';temporary=a.output/'status.tmp'
        temporary.write_text(json.dumps(state,indent=2));os.replace(temporary,target)
        print(json.dumps(dict(phase=phase,**extra)),flush=True)
    try:
        if a.wait_pid:
            save('waiting_for_predecessor',predecessor=a.wait_pid)
            proc=Path(f'/proc/{a.wait_pid}/cmdline')
            while proc.exists():
                try:command=proc.read_bytes()
                except FileNotFoundError:break
                if not command:break  # exited/zombie; completion evidence checked below
                if b'fsdp_convergence_job.py' not in command or str(a.wait_result.parent).encode() not in command:
                    break  # PID no longer identifies this job; never signal it
                time.sleep(15)
            previous=json.loads(a.wait_result.read_text())
            if not previous.get('complete') or not previous.get('best_parameters_restored'):
                raise RuntimeError('Predecessor did not finish correctly; do not start search')
        def idle():
            while True:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True).splitlines()
                readings=[tuple(int(x.strip()) for x in row.split(',')) for row in rows]
                if len(readings)>=4 and all(m<2048 and u<10 for m,u in readings[:4]):return
                time.sleep(15)
        def run(k,name,steps,profile=False):
            save('waiting_for_idle_gpus',next_case=name);idle();save('running_case',case=name)
            output=a.output/name;log=a.output/(name+'.log')
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'benchmarks/compare_fsdp_training.py','--backend','official','--model',str(a.model),
                '--keep-full-layers',str(k),'--memory-budget-gib',str(a.budget_gib),
                '--seq',str(a.seq),'--batch',str(a.batch),'--warmup','3','--steps',str(steps),
                '--reference-check','--output',str(output)]
            if profile:cmd.append('--profile')
            started=time.time()
            with log.open('w') as stream:
                job=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
            raw=log.read_bytes()
            row=dict(keep_full_layers=k,returncode=job.returncode,source=str(output/'result.json'),
                log_sha256=hashlib.sha256(raw).hexdigest(),started=started,finished=time.time())
            if job.returncode:
                if b'out of memory' in raw.lower() or b'OutOfMemoryError' in raw:
                    row['status']='oom';return row
                raise RuntimeError(f'Non-capacity failure in {name}; inspect {log}')
            result=json.loads((output/'result.json').read_text())
            if not result.get('complete'):raise RuntimeError(f'Incomplete worker: {name}')
            row.update(status='passed',result=result,
                result_sha256=hashlib.sha256((output/'result.json').read_bytes()).hexdigest())
            return row
        if a.checkpoint_gate:
            save('waiting_for_idle_gpus',next_case='checkpoint_gate');idle();save('checkpoint_gate')
            gate=a.output/'checkpoint_gate'
            with (a.output/'checkpoint_gate.log').open('w') as stream:
                subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                    'benchmarks/verify_fsdp_checkpoint.py','--cuda','--output',str(gate)],
                    stdout=stream,stderr=subprocess.STDOUT,check=True,
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            result=json.loads((gate/'result.json').read_text())
            if not (result['passed'] and result['cuda_fsdp'] and result['world_size']==4):
                raise RuntimeError('CUDA FSDP checkpoint replay gate did not pass')
            state['checkpoint_gate']=result;save('checkpoint_gate_passed')
        for k in candidates:
            row=run(k,f'screen_k{k}',a.screen_steps)
            state['screen'].append(row);save('screen_case_finished',last_keep=k,last_status=row['status'])
            if k==0 and row['status']!='passed':raise RuntimeError('Baseline does not fit budget')
        selected=select_retention(state['screen'],a.budget_gib)
        best=selected['best_keep_full_layers'];save('screen_complete',selection=selected)
        if best:
            for repeat,order in enumerate(([best,0],[0,best])):
                for k in order:
                    row=run(k,f'confirm_r{repeat}_k{k}',a.confirm_steps,profile=repeat==0)
                    if row['status']!='passed':raise RuntimeError('Winner confirmation failed capacity gate')
                    row['repeat']=repeat;state['confirmation'].append(row);save('confirmation_case_finished')
            comparisons=[]
            for repeat in (0,1):
                pair={x['keep_full_layers']:x['result'] for x in state['confirmation'] if x['repeat']==repeat}
                # Validate two results at identical longer measurement length.
                select_retention([dict(keep_full_layers=k,status='passed',result=r)
                    for k,r in pair.items()],a.budget_gib)
                comparisons.append(dict(repeat=repeat,baseline_seconds=pair[0]['median_step_seconds'],
                    candidate_seconds=pair[best]['median_step_seconds'],
                    throughput_speedup=pair[0]['median_step_seconds']/pair[best]['median_step_seconds']))
            save('finished',complete=True,confirmed_keep_full_layers=best,
                paired_comparisons=comparisons,
                consistent_over_two_percent=all(x['throughput_speedup']>1.02 for x in comparisons))
        else:save('finished',complete=True,confirmed_keep_full_layers=0,consistent_over_two_percent=False)
    except Exception as error:
        save('failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
