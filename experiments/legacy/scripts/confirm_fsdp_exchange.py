"""Fresh-source real8B ABCD/DCBA, including strong native protocol controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.freeze_fsdp_reference import verify_frozen
from scripts.queue_nccl_then_quality import bounded_run,resources_ready
from scripts.summarize_fsdp_memory import audit_result
from training.fsdp_provenance import source_snapshot
from scripts.fsdp_resource_budget import budget_ready


CASES={
    'native_auto_k36': (36,None,'all',None),
    'torch_decoder_auto_k32': (32,'torch','decoder',None),
    'native_simple_k36': (36,None,'all','Simple'),
    'triton_head_simple_k32': (32,'triton','head','Simple'),
}


def command(spec,output,model):
    keep,backend,scope,_=spec
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
        'benchmarks/compare_fsdp_training.py','--backend','official','--model',model,
        '--seq','512','--batch','1','--warmup','3','--steps','12',
        '--keep-full-layers',str(keep),'--checkpoint-every','1',
        '--memory-budget-gib','42','--reference-check','--output',str(output)]
    if backend:cmd+=['--exchange-reduce',backend,'--exchange-scope',scope]
    return cmd


def audit_case(result,spec):
    keep,backend,scope,proto=spec
    signature=audit_result(result,(keep,1,'full'),allow_exchange=True)
    if (result['world_size']!=4 or result['global_parameters']!=8190735360 or
            result['seq']!=512 or result['batch']!=1 or result['steps']!=12 or result['warmup']!=3):
        raise ValueError('Real 8B fixed workload required')
    opt=result['optimization']
    if opt.get('exchange_reduce')!=backend or opt.get('exchange_scope')!=scope:
        raise ValueError('Exchange label mismatch')
    env=result['communication_environment']
    if env!={k:(proto if k=='NCCL_PROTO' else None) for k in
            ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE')}:
        raise ValueError('Unexpected communication environment')
    hashes=result['manifest']['input_sha256_by_rank']
    if len(hashes)!=4 or any(len(h)!=64 for h in hashes):
        raise ValueError('All-rank input digests required')
    counts=result['exchange_callback_counts_by_rank']
    if len(counts)!=4 or (backend and any(not c or any(v<=0 for v in c.values()) for c in counts)):
        raise ValueError('Candidate callback must execute on every rank')
    if not backend and any(counts):raise ValueError('Native control unexpectedly used exchange')
    return signature+(tuple(hashes),json.dumps(result['manifest']['exchange_sources'],sort_keys=True))


def decisions(rows):
    indexed={(r['repeat'],r['case']):r for r in rows}
    output={}
    for candidate,control in (('torch_decoder_auto_k32','native_auto_k36'),
            ('triton_head_simple_k32','native_simple_k36')):
        pairs=[]
        for repeat in (0,1):
            c=indexed[(repeat,candidate)]
            controls=[indexed[(repeat,n)] for n in (control,'native_auto_k36')]
            pairs.append(None if c['status']!='passed' or any(r['status']!='passed' for r in controls)
                else min(r['median_step_seconds'] for r in controls)/c['median_step_seconds'])
        output[candidate]=dict(ratios_vs_stronger_control=pairs,
            over_two_percent_both_orders=all(v is not None and v>1.02 for v in pairs))
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',default='/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
    parser.add_argument('--frozen',type=Path,default=Path('results/frozen_fsdp_reference_v1.json'))
    parser.add_argument('--model-free-margin-gib',type=float,
        help='Opt-in free >=42 GiB allocator + margin; default retains used<2 GiB guard')
    parser.add_argument('--idle-confirmations',type=int,default=1)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.idle_confirmations<1:raise ValueError('Positive idle confirmations required')
    if args.model_free_margin_gib is not None:
        budget_ready([],allocator_gib=42.,free_margin_gib=args.model_free_margin_gib)
    def status(phase,**fields):
        (args.output/'status.json').write_text(json.dumps(dict(phase=phase,**fields),indent=2))
        print(phase,fields,flush=True)
    sources=source_snapshot(Path.cwd())
    extra=[Path(p) for p in ('benchmarks/fsdp_bf16_exchange_adapter.py',
        'benchmarks/bf16_exchange_reduce.py','benchmarks/verify_block_fsdp.py',
        'scripts/confirm_fsdp_exchange.py','scripts/fsdp_resource_budget.py',
        'scripts/queue_nccl_then_quality.py')]
    def extra_hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in extra}
    helpers=extra_hashes()
    def unchanged():
        if source_snapshot(Path.cwd())!=sources or extra_hashes()!=helpers:
            raise ValueError('Source changed during campaign')
    def idle(label):
        status('waiting_for_idle_gpus',case=label,
            model_free_margin_gib=args.model_free_margin_gib,idle_confirmations=args.idle_confirmations)
        samples=[]
        while True:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).splitlines()
            readings=[tuple(int(v.strip()) for v in r.split(',')) for r in rows]
            ready=(resources_ready(readings) if args.model_free_margin_gib is None else
                budget_ready(readings,allocator_gib=42.,free_margin_gib=args.model_free_margin_gib))
            if ready:samples.append(dict(unix_seconds=time.time(),readings=readings[:4]))
            else:samples=[]
            if len(samples)>=args.idle_confirmations:
                (args.output/(label+'_resources.json')).write_text(json.dumps(dict(
                    allocator_budget_gib=42.,free_margin_gib=args.model_free_margin_gib,
                    idle_confirmations=args.idle_confirmations,samples=samples,
                    scope='Launch observations only, not reservation or future-neighbor isolation'),indent=2))
                return
            time.sleep(15)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    for name in ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE',
            'PYTORCH_CUDA_ALLOC_CONF','CUBLAS_WORKSPACE_CONFIG'):
        env.pop(name,None)
    try:
        status('verifying_frozen_inputs')
        frozen=json.loads(args.frozen.read_text());verify_frozen(frozen)
        # Validate checkpoint+retention interaction before measuring the model.
        status('running_checkpoint_exchange_gate')
        gate=args.output/'gate'
        code,timed_out=bounded_run([sys.executable,'-m','scripts.queue_bf16_exchange_gate',
            '--output',str(gate),'--checkpoint-every','1','--keep-full-layers','2'],
            args.output/'gate.log',env,timeout=7200)
        if code or timed_out or json.loads((gate/'status.json').read_text())!={'phase':'finished','passed':True}:
            raise ValueError('Checkpoint+exchange gate failed')
        rows=[];signatures=set()
        for repeat,order in ((0,list(CASES)),(1,list(reversed(CASES)))):
            for name in order:
                label=f'r{repeat}_{name}';idle(label);unchanged()
                status('running_model',case=label)
                run_env=dict(env)
                if CASES[name][3]:run_env['NCCL_PROTO']=CASES[name][3]
                log=args.output/(label+'.log')
                code,timed_out=bounded_run(command(CASES[name],args.output/label,args.model),
                    log,run_env,timeout=3600)
                unchanged()
                row=dict(repeat=repeat,case=name,returncode=code,
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
                if code or timed_out:
                    if timed_out or not any(v in log.read_bytes() for v in
                            (b'CUDA out of memory',b'torch.OutOfMemoryError',b'torch.cuda.OutOfMemoryError')):
                        raise RuntimeError(f'{label}: non-capacity failure, see {log}')
                    row['status']='oom'
                else:
                    path=args.output/label/'result.json';raw=path.read_bytes();result=json.loads(raw)
                    signatures.add(audit_case(result,CASES[name]))
                    row.update(status='passed',source=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                        median_step_seconds=result['median_step_seconds'],
                        peak_allocated_gib=max(r['peak_allocated_gib'] for r in result['records']),
                        peak_reserved_gib=max(r['peak_reserved_gib'] for r in result['records']))
                rows.append(row)
                (args.output/'progress.json').write_text(json.dumps(rows,indent=2))
        if len(signatures)!=1:raise ValueError('Matched source/software/input/workload required')
        verify_frozen(frozen);unchanged()
        report=dict(complete=True,records=rows,decisions=decisions(rows),
            resource_policy=dict(allocator_budget_gib=42.,free_margin_gib=args.model_free_margin_gib,
                idle_confirmations=args.idle_confirmations,
                scope='Shared GPUs, no reservation; nonallocator margin does not guarantee against external growth'),
            implementation_sources=sources,helper_sources=helpers,
            frozen_content_sha256=frozen['content_sha256'],defaults_changed=False,
            scope='Two-order real8B full training steps on shared GPUs, not convergence or statistical significance')
        (args.output/'result.json').write_text(json.dumps(report,indent=2))
        status('finished')
    except Exception as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
