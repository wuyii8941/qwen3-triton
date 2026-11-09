"""Serial tiny-Qwen numerical gates before any real-model installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.queue_nccl_then_quality import bounded_run, resources_ready


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint-every',type=int,default=0)
    parser.add_argument('--keep-full-layers',type=int,default=0)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**fields):
        (args.output/'status.json').write_text(json.dumps(dict(phase=phase,**fields),indent=2))
        print(phase,fields,flush=True)
    files=[Path(p) for p in ('benchmarks/fsdp_bf16_exchange_adapter.py',
        'benchmarks/bf16_exchange_reduce.py','benchmarks/verify_block_fsdp.py')]
    def snapshot():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    initial=snapshot()
    (args.output/'sources.json').write_text(json.dumps(initial,indent=2))
    try:
        for backend,proto in (('torch','auto'),('triton','Simple')):
            label=backend+'_'+proto
            status('waiting_for_idle_gpus',case=label)
            while True:
                rows=subprocess.check_output(['nvidia-smi',
                    '--query-gpu=memory.used,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True).splitlines()
                readings=[tuple(int(v.strip()) for v in row.split(',')) for row in rows]
                if resources_ready(readings,communication=True):break
                time.sleep(15)
            if snapshot()!=initial:raise ValueError('Gate source changed before launch')
            status('running_gate',case=label)
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     CUBLAS_WORKSPACE_CONFIG=':4096:8')
            for name in ('NCCL_PROTO','NCCL_ALGO','NCCL_DEBUG_FILE'):
                env.pop(name,None)
            if proto!='auto':env['NCCL_PROTO']=proto
            command=[sys.executable,'-m','torch.distributed.run','--standalone',
                '--nproc_per_node=4','benchmarks/verify_block_fsdp.py',
                '--backend','official','--bf16','--deterministic','--head-dim','128',
                '--seq','129','--exchange-reduce',backend,
                '--checkpoint-every',str(args.checkpoint_every),
                '--keep-full-layers',str(args.keep_full_layers)]
            log=args.output/(label+'.log')
            code,timed_out=bounded_run(command,log,env,timeout=300)
            if code or timed_out:raise RuntimeError(f'{label}: returncode={code}, timed_out={timed_out}; see {log}')
            if snapshot()!=initial:raise ValueError('Gate source changed during execution')
            records=[]
            for line in log.read_text().splitlines():
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if isinstance(row,dict) and row.get('passed'):records.append(row)
            if (len(records)!=2 or {r['tied'] for r in records}!={False,True} or
                    any(r['world']!=4 or r['steps']!=3 or r['exchange_reduce']!=backend or
                        r['checkpoint_every']!=args.checkpoint_every or
                        r['keep_full_layers']!=args.keep_full_layers or
                        r['exchange_callback_calls']<=0 or not r['deterministic'] for r in records)):
                raise ValueError('Missing complete tied/untied callback evidence')
            (args.output/(label+'_result.json')).write_text(json.dumps(dict(
                passed=True,protocol_request=proto,records=records,source_hashes=initial,
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                scope='D128 S129 four-rank tiny Qwen three-step numerical gate; not real8B performance/quality'),indent=2))
        status('finished',passed=True)
    except Exception as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':main()
