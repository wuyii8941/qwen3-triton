"""Serial confirmation of strongest non-CE and CE candidates, with frozen inputs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

from scripts.freeze_fsdp_reference import verify_frozen
from scripts.summarize_fsdp_memory import summarize as summarize_native,audit_result
from scripts.summarize_fsdp_ce_only import summarize as summarize_ce
from scripts.summarize_fsdp_shape_confirmation import summarize as summarize_confirmation
from training.fsdp_provenance import source_snapshot


def choose_policies(native,ce):
    rows=[r for r in native['records'] if r['status']=='passed']
    baseline=next((r for r in rows if r['case']=='full_k8'),None)
    candidates=[r for r in ce['records'] if r['status']=='passed' and r['policy'][3]>0]
    if baseline is None or not candidates:raise ValueError('Passing baseline and CE candidate required')
    best=min(rows,key=lambda r:r['median_step_seconds'])
    best_ce=min(candidates,key=lambda r:r['median_step_seconds'])
    def native_policy(row):
        p=row['policy'];return (p['keep_full_layers'],p['checkpoint_every'],p['checkpoint_mode'],0)
    if native_policy(baseline)!=(8,2,'full',0):raise ValueError('Baseline policy mismatch')
    policies={'noce_e2_k8':native_policy(baseline)}
    no_ce=f"noce_e{best['policy']['checkpoint_every']}_k{best['policy']['keep_full_layers']}"
    ce_name=f'ce_e{best_ce["policy"][1]}_k{best_ce["policy"][0]}'
    policies[no_ce]=native_policy(best);policies[ce_name]=tuple(best_ce['policy'])
    times={'noce_e2_k8':baseline['median_step_seconds'],no_ce:best['median_step_seconds'],
           ce_name:best_ce['median_step_seconds']}
    return dict(policies=policies,candidate=min(times,key=times.get),
        baseline='noce_e2_k8',no_ce=no_ce,ce=ce_name,
        scope='Screening selection only; cross-round latencies do not establish a causal speedup')


def decisions(report,selection):
    times={r['case']:r['median_step_seconds'] for r in report['records']}
    ce=[];native=[];ce_baseline=[]
    for repeat in (0,1):
        def seconds(name):return times[f'r{repeat}_{name}']
        ce.append(seconds(selection['no_ce'])/seconds(selection['ce']))
        native.append(seconds(selection['baseline'])/seconds(selection['no_ce']))
        ce_baseline.append(seconds(selection['baseline'])/seconds(selection['ce']))
    ce_stable=all(x>1.02 for x in ce+ce_baseline)
    native_stable=all(x>1.02 for x in native)
    return dict(ce_over_best_no_ce_throughput_ratios=ce,
        best_no_ce_over_baseline_throughput_ratios=native,
        ce_over_baseline_throughput_ratios=ce_baseline,
        ce_over_two_percent_in_both_orders=ce_stable,
        no_ce_over_two_percent_in_both_orders=native_stable,
        candidate_for_quality_validation=(selection['ce'] if ce_stable else
            selection['no_ce'] if native_stable else selection['baseline']),
        defaults_changed=False,scope='Preregistered repeatability threshold, not significance or convergence')


def check_runtime(frozen):
    packages={n:importlib.metadata.version(n) for n in ('torch','transformers','triton')}
    if packages!=frozen.get('packages') or platform.python_version()!=frozen.get('python'):
        raise ValueError('Runtime versions differ from frozen reference')


def benchmark_command(workload,policy,output,steps):
    keep,every,mode,chunk=policy
    return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
        'benchmarks/compare_fsdp_training.py','--backend','official','--model',workload['model'],
        '--keep-full-layers',str(keep),'--checkpoint-every',str(every),'--checkpoint-mode',mode,
        '--ce-only-chunk-size',str(chunk),'--memory-budget-gib','42','--batch','1','--seq','2048',
        '--warmup','3','--steps',str(steps),'--reference-check','--output',str(output)]


def verify_numerical_gate(path,every,chunk,keep=1):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    if len(rows)!=2 or {r.get('tied') for r in rows}!={False,True}:
        raise ValueError('Both tied/untied gate records required')
    expected=dict(passed=True,backend='official',dtype='torch.bfloat16',world=4,head_dim=128,
        seq=2048,steps=3,deterministic=True,checkpoint_every=every,checkpoint_mode='full',
        keep_full_layers=keep,loss_chunk_size=0,ce_only_chunk_size=chunk,swiglu=False,
        accumulation=1,gradient_sync='each')
    if any(any(r.get(k)!=v for k,v in expected.items()) for r in rows):
        raise ValueError('Numerical gate does not match declared policy')


def main():
    p=argparse.ArgumentParser()
    for name in ('refinement','refinement-spec','ce-screen','ce-spec','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--predecessor',type=int,required=True)
    p.add_argument('--headroom-refinement',action='store_true',help='Screen all-checkpoint K28/K32/K36 before confirmation')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**extra):
        (a.output/'status.json').write_text(json.dumps(dict(phase=phase,updated=time.time(),**extra),indent=2))
        print(phase,flush=True)
    def idle(name):
        status('waiting_for_idle_gpus',next_case=name)
        while True:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True)
            values=[tuple(int(x.strip()) for x in line.split(',')) for line in text.splitlines()]
            if len(values)>=4 and all(m<2048 and u<10 for m,u in values[:4]):return
            time.sleep(15)
    def run(command,path,env):
        with path.open('x') as stream:
            code=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        return code
    try:
        status('waiting_for_reference',predecessor=a.predecessor)
        predecessor=Path(f'/proc/{a.predecessor}/cmdline')
        while predecessor.exists():
            try:command=predecessor.read_bytes()
            except FileNotFoundError:break
            if b'freeze_fsdp_reference' not in command:break
            time.sleep(15)
        frozen=json.loads(a.reference.read_text());check_runtime(frozen);verify_frozen(frozen)
        if source_snapshot(Path.cwd())!=frozen['implementation_sources']:
            raise ValueError('Implementation changed after reference capture')
        native=summarize_native(a.refinement,json.loads(a.refinement_spec.read_text()),baseline_case='full_k0')
        ce=summarize_ce(a.ce_screen,json.loads(a.ce_spec.read_text()))
        if native['workload_signature']!=ce['workload_signature']:
            raise ValueError('Screen workloads/sources differ; refuse cross-screen selection')
        sample=json.loads((a.refinement/'full_k8/result.json').read_text())
        if sample['manifest']['implementation_sources']!=frozen['implementation_sources']:
            raise ValueError('Screens do not match frozen implementation')
        workload={k:sample[k] for k in ('model','world_size','batch','seq','warmup','steps',
            'memory_budget_gib','precision','attention')}
        if tuple(workload[k] for k in ('world_size','batch','seq','warmup','steps','memory_budget_gib'))!=(4,1,2048,3,6,42):
            raise ValueError('Unexpected screening workload')
        env=dict(os.environ);env.pop('PYTORCH_CUDA_ALLOC_CONF',None);env.pop('CUBLAS_WORKSPACE_CONFIG',None)
        env.update(PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        gate_env=dict(env,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        if a.headroom_refinement:
            anchor=next((r for r in native['records'] if r['case']=='all_k24' and r['status']=='passed'),None)
            if anchor is None:raise ValueError('Feasible all-checkpoint K24 anchor required for headroom search')
            config=json.loads((Path(workload['model'])/'config.json').read_text())
            if config['num_hidden_layers']!=36:raise ValueError('Headroom search scoped to the real 36-layer model')
            root=a.output/'headroom';root.mkdir()
            idle('all_retained_tiny_gate');status('numerical_gate',case='all_retained_tiny')
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'benchmarks/verify_block_fsdp.py','--backend','official','--bf16','--deterministic',
                '--head-dim','128','--seq','2048','--keep-full-layers','2','--checkpoint-every','1']
            path=root/'gate_all_retained.log'
            if run(command,path,gate_env):raise RuntimeError('All-retained numerical gate failed')
            verify_numerical_gate(path,1,0,keep=2)
            added=[]
            for keep in (28,32,36):
                name=f'all_k{keep}';idle(name);status('headroom_screen',case=name)
                log=root/(name+'.log')
                code=run(benchmark_command(workload,(keep,1,'full',0),root/name,6),log,env)
                with (root/'status.log').open('a') as stream:stream.write(f'{name} returncode={code}\n')
                log_bytes=log.read_bytes()
                row=dict(case=name,returncode=code,policy=dict(keep_full_layers=keep,checkpoint_every=1,checkpoint_mode='full'),
                    log_source=str(log),log_sha256=hashlib.sha256(log_bytes).hexdigest())
                if code:
                    if not any(s in log_bytes for s in (b'torch.OutOfMemoryError',b'CUDA out of memory')):
                        raise RuntimeError(f'Non-capacity headroom failure: {name}')
                    row.update(status='oom')
                else:
                    path=root/name/'result.json';raw=path.read_bytes();r=json.loads(raw)
                    if list(audit_result(r,(keep,1,'full')))!=native['workload_signature']:
                        raise ValueError('Headroom result source/input/workload mismatch')
                    row.update(status='passed',source=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                        median_step_seconds=r['median_step_seconds'],tokens_per_second=r['global_tokens_per_second'],
                        peak_allocated_gib=max(x['peak_allocated_gib'] for x in r['records']),
                        peak_reserved_gib=max(x['peak_reserved_gib'] for x in r['records']),initial_hf_parity=r['initial_hf_parity'])
                added.append(row)
            native['records'].extend(added)
            native['screening_winner']=min((r for r in native['records'] if r['status']=='passed'),
                key=lambda r:r['median_step_seconds'])['case']
            native['scope']='Initial refinement plus headroom screen; selection only, not repeated confirmation'
            (root/'result.json').write_text(json.dumps(dict(complete=True,records=added),indent=2)+'\n')
        selection=choose_policies(native,ce)
        workload['steps']=12
        spec=dict(policies=selection['policies'],candidate=selection['candidate'],workload=workload)
        for name,value in (('selection',selection),('spec',spec),('native_screen',native),('ce_screen',ce)):
            (a.output/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        for name,(keep,every,mode,chunk) in selection['policies'].items():
            idle('gate_'+name);status('numerical_gate',case=name)
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'benchmarks/verify_block_fsdp.py','--backend','official','--bf16','--deterministic',
                '--head-dim','128','--seq','2048','--keep-full-layers','1','--checkpoint-every',str(every),
                '--checkpoint-mode',mode,'--ce-only-chunk-size',str(chunk)]
            path=a.output/f'gate_{name}.log'
            if run(command,path,gate_env):raise RuntimeError(f'Numerical gate failed: {name}')
            verify_numerical_gate(path,every,chunk)
        names=list(selection['policies'])
        order=[(0,n) for n in names]+[(1,n) for n in reversed(names)]
        for repeat,name in order:
            case=f'r{repeat}_{name}';keep,every,mode,chunk=selection['policies'][name]
            idle(case);status('running',case=case)
            command=benchmark_command(workload,selection['policies'][name],a.output/case,12)
            code=run(command,a.output/f'{case}.log',env)
            with (a.output/'status.log').open('a') as stream:stream.write(f'{case} returncode={code}\n')
            if code:raise RuntimeError(f'Confirmation failed: {case}; do not skip a failed candidate')
        report=summarize_confirmation(a.output,spec)
        check_runtime(frozen);verify_frozen(frozen)
        report['decision']=decisions(report,selection)
        report['reference_content_sha256']=frozen['content_sha256']
        report['supervisor_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        status('finished',decision=report['decision'])
    except Exception as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
