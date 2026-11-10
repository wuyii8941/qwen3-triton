"""Additional 40GiB real8B Pareto study; never replaces the parked 42GiB run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.confirm_fsdp_exchange import command as base_command,audit_case
from scripts.audit_fsdp_exchange_confirmation import callback_contract
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.fsdp_resource_budget import budget_ready
from scripts.screen_fsdp_1p7b import park_waiting_queue
from training.fsdp_provenance import source_snapshot


CASES={
    'native_auto_k32':(32,None,'all',None),
    'native_auto_k34':(34,None,'all',None),
    'native_simple_k32':(32,None,'all','Simple'),
    'native_simple_k34':(34,None,'all','Simple'),
    'torch_decoder_auto_k28':(28,'torch','decoder',None),
    'triton_head_simple_k26':(26,'triton','head','Simple'),
}


def command(spec,output):
    cmd=base_command(spec,output,'/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
    cmd[cmd.index('--memory-budget-gib')+1]='40'
    return cmd


def verify_gates(root):
    for name,backend,proto in (('torch_auto','torch','auto'),('triton_Simple','triton','Simple')):
        path=root/(name+'_result.json');record=json.loads(path.read_text())
        if record.get('passed') is not True or record['protocol_request']!=proto:
            raise ValueError('Passing matching numerical gate required')
        if hashlib.sha256((root/(name+'.log')).read_bytes()).hexdigest()!=record['log_sha256']:
            raise ValueError('Changed gate log')
        if set(record['source_hashes'])!={'benchmarks/fsdp_bf16_exchange_adapter.py',
                'benchmarks/bf16_exchange_reduce.py','benchmarks/verify_block_fsdp.py'}:
            raise ValueError('Complete numerical gate source identity required')
        for filename,digest in record['source_hashes'].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest:raise ValueError('Gate source changed')
        rows=record['records']
        if (len(rows)!=2 or {r['tied'] for r in rows}!={False,True} or any(
                not r['passed'] or r['exchange_reduce']!=backend or r['world']!=4 or
                r['head_dim']!=128 or r['seq']!=129 or r['steps']!=3 or
                r['checkpoint_every']!=1 or r['keep_full_layers']!=2 or
                not r['deterministic'] or r['exchange_callback_calls']<=0 for r in rows)):
            raise ValueError('Incomplete tied/untied numerical gate')


def decisions(rows):
    by={(r['repeat'],r['case']):r for r in rows};output={}
    for candidate,spec in CASES.items():
        if not spec[1]:continue
        ratios=[]
        for repeat in (0,1):
            row=by[(repeat,candidate)]
            controls=[by[(repeat,n)] for n,s in CASES.items() if not s[1]
                and by[(repeat,n)]['status']=='passed']
            matched=any(CASES[c['case']][3]==spec[3] for c in controls)
            ratios.append(min(c['median_step_seconds'] for c in controls)/row['median_step_seconds']
                if row['status']=='passed' and matched else None)
        output[candidate]=dict(ratios_vs_best_feasible_native=ratios,
            over_two_percent_both_orders=all(v is not None and v>1.02 for v in ratios))
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--park-pid',type=int,required=True);parser.add_argument('--park-status',type=Path,required=True)
    parser.add_argument('--gates',type=Path,default=Path('artifacts/fsdp_exchange_model_confirmation_v2/gate'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**fields):
        (args.output/'status.json').write_text(json.dumps(dict(phase=phase,**fields),indent=2))
        print(phase,fields,flush=True)
    def interrupted(signum,frame):raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    sources=source_snapshot(Path.cwd())
    helpers=[Path(p) for p in ('scripts/screen_fsdp_exchange_40.py','scripts/confirm_fsdp_exchange.py',
        'scripts/audit_fsdp_exchange_confirmation.py','scripts/fsdp_resource_budget.py',
        'scripts/screen_fsdp_1p7b.py','scripts/diagnose_fsdp_1p7b_repeat.py',
        'scripts/summarize_fsdp_memory.py','benchmarks/fsdp_bf16_exchange_adapter.py',
        'benchmarks/bf16_exchange_reduce.py','benchmarks/verify_block_fsdp.py')]
    def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}
    helper_sources=hashes()
    def unchanged():
        if source_snapshot(Path.cwd())!=sources or hashes()!=helper_sources:raise ValueError('Study source changed')
    frozen=json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0,1,2,3',
        PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    for key in ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE',
            'CUBLAS_WORKSPACE_CONFIG','PYTORCH_CUDA_ALLOC_CONF'):
        env.pop(key,None)
    def idle(label):
        status('waiting_for_idle_gpus',case=label,allocator_budget_gib=40,free_margin_gib=2.5)
        samples=[]
        while True:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).splitlines()
            values=[tuple(int(v.strip()) for v in row.split(',')) for row in rows]
            if budget_ready(values,allocator_gib=40.,free_margin_gib=2.5):
                samples.append(dict(unix_seconds=time.time(),readings=values[:4]))
            else:samples=[]
            if len(samples)==3:
                (args.output/(label+'_resources.json')).write_text(json.dumps(dict(allocator_budget_gib=40.,
                    free_margin_gib=2.5,idle_confirmations=3,samples=samples),indent=2));return
            time.sleep(15)
    try:
        with park_waiting_queue(args.park_pid,args.park_status):
            status('verifying_frozen_inputs_and_gates');verify_gates(args.gates);verify_frozen(frozen)
            rows=[];signatures=set()
            for repeat,order in ((0,list(CASES)),(1,list(reversed(CASES)))):
                for name in order:
                    label=f'r{repeat}_{name}';idle(label);unchanged();status('running_model',case=label)
                    run_env=dict(env)
                    if CASES[name][3]:run_env['NCCL_PROTO']=CASES[name][3]
                    log=args.output/(label+'.log')
                    code=run_owned(command(CASES[name],args.output/label),log,run_env,timeout=3600)
                    unchanged();row=dict(repeat=repeat,case=name,returncode=code,
                        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
                    if code:
                        if not any(t in log.read_bytes() for t in
                                (b'CUDA out of memory',b'torch.OutOfMemoryError',b'torch.cuda.OutOfMemoryError')):
                            raise RuntimeError(f'{label}: non-capacity failure')
                        row['status']='oom'
                    else:
                        path=args.output/label/'result.json';raw=path.read_bytes();result=json.loads(raw)
                        if result['memory_budget_gib']!=40:raise ValueError('Wrong memory budget')
                        signatures.add(audit_case(result,CASES[name]));callback_contract(result,CASES[name])
                        row.update(status='passed',source=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                            median_step_seconds=result['median_step_seconds'],
                            peak_allocated_gib=max(r['peak_allocated_gib'] for r in result['records']),
                            peak_reserved_gib=max(r['peak_reserved_gib'] for r in result['records']))
                    rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2))
            if len(signatures)!=1 or any(r['status']!='passed' for r in rows if r['case']=='native_auto_k32'):
                raise ValueError('Passing matched controls required')
            verify_frozen(frozen);unchanged()
            (args.output/'result.json').write_text(json.dumps(dict(complete=True,memory_budget_gib=40,
                case_specs=CASES,records=rows,decisions=decisions(rows),implementation_sources=sources,
                helper_sources=helper_sources,frozen_content_sha256=frozen['content_sha256'],
                defaults_changed=False,original_42gib_study_replaced=False,
                scope='Additional real8B 40GiB budget Pareto study,12 steps/order; not pooled with42GiB or convergence'),indent=2))
        status('finished',original_42gib_supervisor_resumed=True)
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
