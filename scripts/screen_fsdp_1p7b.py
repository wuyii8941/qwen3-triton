"""Serial two-card 1.7B study while the four-card queue is only waiting.

Only the verified waiting supervisor is stopped, never its GPU workers or any
unrelated process. Finally resumes that exact process identity on all exits.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.freeze_fsdp_reference import snapshot,verify_frozen,weight_path
from scripts.queue_nccl_then_quality import bounded_run
from scripts.summarize_fsdp_memory import audit_result
from training.fsdp_provenance import source_snapshot


POLICIES={'native_k0':(0,0,'full'),'native_k14':(14,0,'full'),
          'native_k28':(28,0,'full'),'checkpoint_k28':(28,1,'full')}


def identity(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if not any(cmd[i:i+2]==[b'-m',b'scripts.confirm_fsdp_exchange'] for i in range(len(cmd)-1)):
        raise ValueError('PID is not the expected model supervisor')
    if Path(f'/proc/{pid}/cwd').resolve()!=Path.cwd():raise ValueError('Wrong supervisor workspace')
    return fields[19],fields[0]


def has_children(pid):
    result=subprocess.run(['ps','--ppid',str(pid),'-o','pid='],text=True,capture_output=True)
    if result.returncode not in (0,1):raise RuntimeError('Could not inspect supervisor children')
    return bool(result.stdout.strip())


@contextmanager
def park_waiting_queue(pid,status_file):
    initial,state=identity(pid)
    if state in ('T','t'):raise ValueError('Supervisor was already stopped')
    if json.loads(status_file.read_text()).get('phase')!='waiting_for_idle_gpus' or has_children(pid):
        raise ValueError('Do not interrupt active GPU work')
    os.kill(pid,signal.SIGSTOP)
    try:
        deadline=time.monotonic()+2
        while identity(pid)[1] not in ('T','t'):
            if time.monotonic()>deadline:raise RuntimeError('Supervisor did not stop')
            time.sleep(.01)
        # Freeze first, then recheck the launch race. Resume on any mismatch.
        if identity(pid)[0]!=initial or has_children(pid) or \
                json.loads(status_file.read_text()).get('phase')!='waiting_for_idle_gpus':
            raise ValueError('Supervisor began work before it could be parked')
        yield
    finally:
        try:current,_=identity(pid)
        except FileNotFoundError:current=None
        if current==initial:os.kill(pid,signal.SIGCONT)


def resource_ready(readings):
    return len(readings)==2 and all(free>=32768 and util<10 for free,util in readings)


def worker_command(policy,output,model):
    keep,every,mode=policy
    return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        'benchmarks/compare_fsdp_training.py','--backend','official','--model',str(model),
        '--seq','512','--batch','1','--warmup','3','--steps','12','--memory-budget-gib','24',
        '--keep-full-layers',str(keep),'--checkpoint-every',str(every),
        '--checkpoint-mode',mode,'--reference-check','--output',str(output)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--park-pid',type=int,required=True)
    parser.add_argument('--park-status',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('/data1/tzh/models/Qwen/Qwen3-1.7B'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**fields):
        (args.output/'status.json').write_text(json.dumps(dict(phase=phase,**fields),indent=2))
        print(phase,fields,flush=True)
    sources=source_snapshot(Path.cwd())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,2',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    for key in ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE',
            'PYTORCH_CUDA_ALLOC_CONF','CUBLAS_WORKSPACE_CONFIG'):
        env.pop(key,None)
    def idle(label):
        status('waiting_for_two_idle_gpus',case=label,physical_gpus=[0,2])
        while True:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).splitlines()
            values=[tuple(int(v.strip()) for v in r.split(',')) for r in rows]
            if len(values)>=3 and resource_ready([values[0],values[2]]):return
            time.sleep(15)
    def unchanged():
        if source_snapshot(Path.cwd())!=sources:raise ValueError('Implementation changed during experiment')
    try:
        with park_waiting_queue(args.park_pid,args.park_status):
            status('parked_waiting_supervisor',park_pid=args.park_pid)
            config=json.loads((args.model/'config.json').read_text())
            if config['num_hidden_layers']!=28 or config['hidden_size']!=2048:
                raise ValueError('Expected Qwen3-1.7B architecture')
            index=args.model/'model.safetensors.index.json'
            mapping=json.loads(index.read_text())['weight_map']
            paths={'model/config.json':args.model/'config.json','model/index.json':index,
                'corpus/train.bin':Path('artifacts/convergence_corpus_v1/train.bin')}
            paths.update({'model/'+name:weight_path(args.model,name) for name in set(mapping.values())})
            frozen=dict(complete=True,**snapshot(paths))
            (args.output/'frozen_inputs.json').write_text(json.dumps(frozen,indent=2))
            for every in (0,1):
                label=f'gate_e{every}';idle(label);unchanged();status('running_gate',case=label)
                gate_env=dict(env,CUBLAS_WORKSPACE_CONFIG=':4096:8')
                cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                    'benchmarks/verify_block_fsdp.py','--backend','official','--bf16','--deterministic',
                    '--head-dim','128','--seq','129','--keep-full-layers','2',
                    '--checkpoint-every',str(every)]
                code,timeout=bounded_run(cmd,args.output/(label+'.log'),gate_env,timeout=300)
                if code or timeout:raise RuntimeError(f'{label} failed, see raw log')
                records=[]
                for line in (args.output/(label+'.log')).read_text().splitlines():
                    try:row=json.loads(line)
                    except json.JSONDecodeError:continue
                    if isinstance(row,dict) and row.get('passed'):records.append(row)
                if (len(records)!=2 or {r['tied'] for r in records}!={False,True} or
                        any(r['world']!=2 or r['steps']!=3 or r['checkpoint_every']!=every for r in records)):
                    raise ValueError('Incomplete two-card gate')
            rows=[];signatures=set()
            for repeat,order in ((0,list(POLICIES)),(1,list(reversed(POLICIES)))):
                for name in order:
                    label=f'r{repeat}_{name}';idle(label);unchanged();status('running_model',case=label)
                    log=args.output/(label+'.log')
                    code,timeout=bounded_run(worker_command(POLICIES[name],args.output/label,args.model),
                        log,env,timeout=1800)
                    unchanged()
                    row=dict(case=name,repeat=repeat,returncode=code,
                        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
                    if code or timeout:
                        if timeout or not any(t in log.read_bytes() for t in
                                (b'CUDA out of memory',b'torch.OutOfMemoryError',b'torch.cuda.OutOfMemoryError')):
                            raise RuntimeError(f'{label}: non-capacity failure')
                        row['status']='oom'
                    else:
                        path=args.output/label/'result.json';raw=path.read_bytes();result=json.loads(raw)
                        signatures.add(audit_result(result,POLICIES[name]))
                        if result['world_size']!=2 or result['seq']!=512 or result['memory_budget_gib']!=24:
                            raise ValueError('Wrong workload')
                        row.update(status='passed',source=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                            median_step_seconds=result['median_step_seconds'],
                            global_parameters=result['global_parameters'],
                            peak_allocated_gib=max(r['peak_allocated_gib'] for r in result['records']))
                    rows.append(row)
                    (args.output/'progress.json').write_text(json.dumps(rows,indent=2))
            if len(signatures)!=1:raise ValueError('Matched workload/source/software required')
            baseline={r['repeat']:r for r in rows if r['case']=='native_k0'}
            if any(r['status']!='passed' for r in baseline.values()):raise ValueError('Passing controls required')
            for r in rows:
                if r['status']=='passed':r['throughput_ratio_vs_k0']=baseline[r['repeat']]['median_step_seconds']/r['median_step_seconds']
            verify_frozen(frozen);unchanged()
            (args.output/'result.json').write_text(json.dumps(dict(complete=True,records=rows,
                physical_gpus=[0,2],implementation_sources=sources,
                frozen_content_sha256=frozen['content_sha256'],defaults_changed=False,
                scope='Real1.7B two-card S512 screening; no convergence or cross-8B speedup claim'),indent=2))
        status('finished',waiting_supervisor_resumed=True)
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
