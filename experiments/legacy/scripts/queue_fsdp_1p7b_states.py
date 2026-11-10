"""Safely serialize the full-state 1.7B gate before resuming waiting 8B."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.screen_fsdp_1p7b import park_waiting_queue,resource_ready


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--park-pid',type=int,required=True);parser.add_argument('--park-status',type=Path,required=True)
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.timeout_seconds<1:raise ValueError('Positive verification timeout required')
    def status(phase,**fields):
        (args.output/'status.json').write_text(json.dumps(dict(phase=phase,**fields),indent=2))
        print(phase,fields,flush=True)
    def interrupted(signum,frame):raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    frozen=json.loads(Path('artifacts/fsdp_1p7b_two_card_screen_v1/frozen_inputs.json').read_text())
    worker=Path('benchmarks/verify_fsdp_1p7b_states.py');worker_hash=hashlib.sha256(worker.read_bytes()).hexdigest()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,2',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    for key in ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','PYTORCH_CUDA_ALLOC_CONF'):
        env.pop(key,None)
    try:
        with park_waiting_queue(args.park_pid,args.park_status):
            status('verifying_frozen_inputs');verify_frozen(frozen)
            status('waiting_for_two_idle_gpus')
            while True:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True).splitlines()
                values=[tuple(int(v.strip()) for v in r.split(',')) for r in rows]
                if len(values)>=3 and resource_ready([values[0],values[2]]):break
                time.sleep(15)
            status('running_full_state_gate')
            log=args.output/'worker.log'
            code=run_owned([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                str(worker),'--output',str(args.output/'gate')],log,env,timeout=args.timeout_seconds)
            if code:raise RuntimeError('Full-state comparison failed; see preserved worker log')
            if hashlib.sha256(worker.read_bytes()).hexdigest()!=worker_hash:raise ValueError('Worker changed')
            summary=json.loads((args.output/'gate/result.json').read_text())
            if not summary['passed'] or summary['worker_sha256']!=worker_hash:
                raise ValueError('Incomplete gate summary')
            raw_reports=[]
            for rank in (0,1):
                path=args.output/'gate'/f'rank{rank}_result.json';raw=path.read_bytes();report=json.loads(raw)
                if (report['rank']!=rank or report['world']!=2 or report['real_parameters']!=1720574976 or
                        report['parameter_entries_per_rank']!=310 or report['steps_per_phase']!=3 or
                        report['worker_sha256']!=worker_hash or report['passed'] is not True):
                    raise ValueError('All-rank full-state evidence required')
                phases=report['phases'];base=phases['native_a']
                if set(phases)!={'native_a','native_b','keep28','checkpoint28'} or len(base)!=3:
                    raise ValueError('Incomplete phase/step coverage')
                for records in phases.values():
                    if records!=base or any(len(r['gradients'])!=310 or len(r['parameters'])!=310 or
                            len(r['optimizer_states'])!=310 for r in records):
                        raise ValueError('State evidence is not exactly equal or complete')
                raw_reports.append(dict(rank=rank,path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
            verify_frozen(frozen)
            (args.output/'result.json').write_text(json.dumps(dict(passed=True,world=2,
                real_parameters=1720574976,steps=3,parameter_entries_per_rank=310,
                execution_timeout_seconds=args.timeout_seconds,
                raw_reports=raw_reports,worker_sha256=worker_hash,
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                frozen_content_sha256=frozen['content_sha256'],
                scope=summary['scope'],defaults_changed=False),indent=2))
        status('finished',waiting_supervisor_resumed=True)
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
