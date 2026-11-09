"""Real-scale deterministic repeat/retention diagnostic, separate from timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.audit_fsdp_exchange_confirmation import loss_comparison
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.screen_fsdp_1p7b import park_waiting_queue,resource_ready
from scripts.summarize_fsdp_memory import audit_result
from training.fsdp_provenance import source_snapshot


def run_owned(command,log,env,*,timeout=600):
    """On interruption, reap the owned launcher before resuming parked work."""
    with log.open('x') as stream:
        process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=15)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--park-pid',type=int,required=True)
    parser.add_argument('--park-status',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,default=Path('artifacts/fsdp_1p7b_two_card_screen_v1/frozen_inputs.json'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**fields):
        (args.output/'status.json').write_text(json.dumps(dict(phase=phase,**fields),indent=2))
        print(phase,fields,flush=True)
    def interrupted(signum,frame):raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    sources=source_snapshot(Path.cwd())
    wrapper=Path('benchmarks/deterministic_fsdp_comparison.py')
    wrapper_hash=hashlib.sha256(wrapper.read_bytes()).hexdigest()
    frozen=json.loads(args.frozen.read_text())
    def unchanged():
        if source_snapshot(Path.cwd())!=sources or hashlib.sha256(wrapper.read_bytes()).hexdigest()!=wrapper_hash:
            raise ValueError('Implementation changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,2',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    for key in ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','PYTORCH_CUDA_ALLOC_CONF'):
        env.pop(key,None)
    try:
        with park_waiting_queue(args.park_pid,args.park_status):
            status('verifying_frozen_inputs');verify_frozen(frozen)
            runs={};provenance={};signatures=set()
            for name,keep,every in (('native_a',0,0),('native_b',0,0),('keep28',28,0),('checkpoint28',28,1)):
                status('waiting_for_two_idle_gpus',case=name)
                while True:
                    rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
                        '--format=csv,noheader,nounits'],text=True).splitlines()
                    values=[tuple(int(v.strip()) for v in r.split(',')) for r in rows]
                    if len(values)>=3 and resource_ready([values[0],values[2]]):break
                    time.sleep(15)
                unchanged();status('running_diagnostic',case=name)
                command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                    str(wrapper),'--backend','official','--model','/data1/tzh/models/Qwen/Qwen3-1.7B',
                    '--seq','512','--batch','1','--warmup','1','--steps','3',
                    '--memory-budget-gib','24','--keep-full-layers',str(keep),
                    '--checkpoint-every',str(every),'--reference-check','--output',str(args.output/name)]
                log=args.output/(name+'.log')
                if run_owned(command,log,env):raise RuntimeError(f'{name} failed; no fallback or tolerance change')
                unchanged();path=args.output/name/'result.json';raw=path.read_bytes();result=json.loads(raw)
                signature=audit_result(result,(keep,every,'full'))
                diagnostic=result.get('deterministic_diagnostic',{})
                if (result['global_parameters']!=1720574976 or result['world_size']!=2 or
                        result['seq']!=512 or result['batch']!=1 or result['warmup']!=1 or
                        result['steps']!=3 or diagnostic.get('enabled') is not True or
                        diagnostic.get('cublas_workspace')!=':4096:8' or
                        diagnostic['source_hashes'].get(wrapper.name)!=wrapper_hash):
                    raise ValueError('Wrong diagnostic contract')
                signatures.add(signature+(tuple(result['manifest']['input_sha256_by_rank']),))
                runs[name]=result
                provenance[name]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
            if len(signatures)!=1:raise ValueError('Diagnostic runs do not match')
            checks={name:loss_comparison(result,runs['native_a']) for name,result in runs.items()
                if name!='native_a'}
            verify_frozen(frozen);unchanged()
            (args.output/'result.json').write_text(json.dumps(dict(complete=True,checks=checks,
                native_repeat_losses_exact=all(a['loss']==b['loss'] for a,b in
                    zip(runs['native_a']['records'],runs['native_b']['records'])),
                losses={name:[r['loss'] for r in result['records']] for name,result in runs.items()},
                provenance=provenance,implementation_sources=sources,wrapper_sha256=wrapper_hash,
                frozen_content_sha256=frozen['content_sha256'],defaults_changed=False,
                scope='Four real1.7B deterministic updates, mean loss only; not full gradient/moment parity, 15-step replay, convergence or performance'),indent=2))
        status('finished',waiting_supervisor_resumed=True)
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
