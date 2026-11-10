"""Real1.7B fixed-global-batch ABC/CBA, never pool with globalbatch2."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.audit_fsdp_accumulation import audit_result
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.screen_fsdp_1p7b import park_waiting_queue,resource_ready
from training.fsdp_provenance import source_snapshot


SPECS={'large_batch':(2,1,'each'),'accum_each':(1,2,'each'),'accum_defer':(1,2,'defer')}


def command(spec,output):
    batch,accum,sync=spec
    return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        'benchmarks/compare_fsdp_training.py','--backend','official',
        '--model','/data1/tzh/models/Qwen/Qwen3-1.7B','--seq','512',
        '--batch',str(batch),'--accumulation',str(accum),'--gradient-sync',sync,
        '--keep-full-layers','28','--memory-budget-gib','24','--warmup','3','--steps','12',
        '--reference-check','--output',str(output)]


def audit_model(result,spec):
    batch,accum,sync=spec
    signature=audit_result(result,batch=batch,accumulation=accum,gradient_sync=sync,policy=(28,0,'full'))
    if (result['global_parameters']!=1720574976 or result['world_size']!=2 or
            result['effective_global_batch']!=4 or result['seq']!=512 or
            result['memory_budget_gib']!=24 or result['steps']!=12 or result['warmup']!=3):
        raise ValueError('Fixed real1.7B world2 globalbatch4 workload required')
    if len(result['manifest'].get('input_sha256_by_rank',[]))!=2:
        raise ValueError('Both rank input digests required')
    if any(result['communication_environment'].values()) or any(result['exchange_callback_counts_by_rank']):
        raise ValueError('No custom protocol or exchange allowed in batch study')
    return signature


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--park-pid',type=int,required=True);parser.add_argument('--park-status',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**fields):
        (args.output/'status.json').write_text(json.dumps(dict(phase=phase,**fields),indent=2))
        print(phase,fields,flush=True)
    def interrupted(signum,frame):raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    sources=source_snapshot(Path.cwd())
    helpers=[Path(p) for p in ('scripts/screen_fsdp_1p7b_accumulation.py','scripts/audit_fsdp_accumulation.py',
        'scripts/screen_fsdp_1p7b.py','scripts/diagnose_fsdp_1p7b_repeat.py','benchmarks/verify_block_fsdp.py')]
    def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}
    initial_helpers=hashes()
    frozen=json.loads(Path('artifacts/fsdp_1p7b_two_card_screen_v1/frozen_inputs.json').read_text())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,2',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    for key in ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE',
            'CUBLAS_WORKSPACE_CONFIG','PYTORCH_CUDA_ALLOC_CONF'):
        env.pop(key,None)
    def unchanged():
        if source_snapshot(Path.cwd())!=sources or hashes()!=initial_helpers:
            raise ValueError('Study source changed')
    def idle(case):
        status('waiting_for_two_idle_gpus',case=case)
        while True:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).splitlines()
            values=[tuple(int(v.strip()) for v in r.split(',')) for r in rows]
            if len(values)>=3 and resource_ready([values[0],values[2]]):return
            time.sleep(15)
    try:
        with park_waiting_queue(args.park_pid,args.park_status):
            status('verifying_frozen_inputs');verify_frozen(frozen)
            for name,(_,accum,sync) in SPECS.items():
                idle('gate_'+name);unchanged();status('running_gate',case=name)
                gate_env=dict(env,CUBLAS_WORKSPACE_CONFIG=':4096:8')
                gate=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                    'benchmarks/verify_block_fsdp.py','--backend','official','--bf16','--deterministic',
                    '--head-dim','128','--seq','129','--keep-full-layers','2',
                    '--accumulation',str(accum),'--gradient-sync',sync]
                log=args.output/('gate_'+name+'.log')
                if run_owned(gate,log,gate_env):raise RuntimeError(f'{name} numerical gate failed')
                records=[]
                for line in log.read_text().splitlines():
                    try:row=json.loads(line)
                    except json.JSONDecodeError:continue
                    if isinstance(row,dict) and row.get('passed'):records.append(row)
                if (len(records)!=2 or {r['tied'] for r in records}!={False,True} or
                        any(r['world']!=2 or r['steps']!=3 or r['head_dim']!=128 or r['seq']!=129 or
                            r['accumulation']!=accum or r['gradient_sync']!=sync or not r['deterministic'] for r in records)):
                    raise ValueError('Incomplete same-microbatch native-reference numerical gate')
            rows=[];signatures=set()
            for repeat,order in ((0,list(SPECS)),(1,list(reversed(SPECS)))):
                for name in order:
                    label=f'r{repeat}_{name}';idle(label);unchanged();status('running_model',case=label)
                    log=args.output/(label+'.log')
                    code=run_owned(command(SPECS[name],args.output/label),log,env,timeout=1800)
                    unchanged();row=dict(repeat=repeat,case=name,returncode=code,
                        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
                    if code:
                        if not any(t in log.read_bytes() for t in
                                (b'CUDA out of memory',b'torch.OutOfMemoryError',b'torch.cuda.OutOfMemoryError')):
                            raise RuntimeError(f'{label}: non-capacity failure')
                        row['status']='oom'
                    else:
                        path=args.output/label/'result.json';raw=path.read_bytes();result=json.loads(raw)
                        signatures.add(audit_model(result,SPECS[name]))
                        row.update(status='passed',source=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                            median_step_seconds=result['median_step_seconds'],
                            peak_allocated_gib=max(r['peak_allocated_gib'] for r in result['records']),
                            tokens_per_second=result['global_tokens_per_second'])
                    rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2))
            if len(signatures)!=1:raise ValueError('Matched global batch/source/software/all-rank inputs required')
            baselines={r['repeat']:r for r in rows if r['case']=='large_batch'}
            if any(r['status']!='passed' for r in baselines.values()):raise ValueError('Passing B2/A1 controls required')
            for row in rows:
                if row['status']=='passed':row['ratio_vs_large_batch']=baselines[row['repeat']]['median_step_seconds']/row['median_step_seconds']
            verify_frozen(frozen);unchanged()
            (args.output/'result.json').write_text(json.dumps(dict(complete=True,records=rows,
                specs=SPECS,implementation_sources=sources,helper_sources=initial_helpers,
                frozen_content_sha256=frozen['content_sha256'],defaults_changed=False,
                scope='Real1.7B world2 B1/A2 vs B2/A1, globalbatch4, S512,24GiB. Same ordered inputs/objective, not BF16 bitwise or long-quality equivalence. No comparison against prior globalbatch2.'),indent=2))
        status('finished',waiting_supervisor_resumed=True)
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
