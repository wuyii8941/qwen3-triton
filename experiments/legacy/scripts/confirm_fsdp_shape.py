"""Serial, gated screen-to-confirmation supervisor; no new training policy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.summarize_fsdp_memory import summarize as summarize_screen
from scripts.summarize_fsdp_shape_confirmation import summarize as summarize_confirmation


def select_policies(report):
    passed=[r for r in report['records'] if r['status']=='passed']
    native=[r for r in passed if r['policy']['checkpoint_every']==0]
    checkpoint=[r for r in passed if r['policy']['checkpoint_every']>0]
    if not native or not checkpoint:raise ValueError('Passing native and checkpoint policies required')
    chosen=[min(group,key=lambda r:r['median_step_seconds']) for group in (native,checkpoint)]
    policies={r['case']:tuple(r['policy'][k] for k in
        ('keep_full_layers','checkpoint_every','checkpoint_mode')) for r in chosen}
    return policies,chosen[1]['case']


def verify_gate(path, every, seq):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    if len(rows)!=2 or {r['tied'] for r in rows}!={False,True}:
        raise ValueError('Both tied/untied gate records required')
    for r in rows:
        expected=dict(passed=True,backend='official',dtype='torch.bfloat16',world=4,
            head_dim=128,seq=seq,steps=3,deterministic=True,checkpoint_every=every,
            checkpoint_mode='full',keep_full_layers=1,loss_chunk_size=0,swiglu=False)
        if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('Mismatched numerical gate')


def main():
    p=argparse.ArgumentParser();p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--case-spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--predecessor',type=int,default=0);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**extra):
        (args.output/'status.json').write_text(json.dumps(dict(phase=phase,updated=time.time(),**extra),indent=2))
        print(phase,flush=True)
    def idle(name):
        status('waiting_for_idle_gpus',next_case=name)
        while True:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True)
            values=[tuple(int(x.strip()) for x in line.split(',')) for line in text.splitlines()]
            if len(values)>=4 and all(m<2048 and u<10 for m,u in values[:4]):return
            time.sleep(15)
    try:
        status('waiting_for_screen',predecessor=args.predecessor)
        if args.predecessor:
            path=Path(f'/proc/{args.predecessor}/cmdline')
            while path.exists():
                try:command=path.read_bytes()
                except FileNotFoundError:break
                if str(args.screen.resolve()).encode() not in command:break
                time.sleep(15)
        report=summarize_screen(args.screen,json.loads(args.case_spec.read_text()),baseline_case='native_k16')
        policies,candidate=select_policies(report)
        initial=json.loads((args.screen/next(iter(policies))/'result.json').read_text())
        workload={k:initial[k] for k in ('model','world_size','batch','seq','warmup','steps',
            'memory_budget_gib','precision','attention')}
        if tuple(workload[k] for k in ('world_size','batch','seq','warmup','steps','memory_budget_gib'))!=(4,1,1024,3,6,42):
            raise ValueError('This supervisor is scoped to the preregistered S1024 screen')
        for every in (0,2):verify_gate(args.screen/f'gate_every{every}_deterministic.log',every,1024)
        workload['steps']=12
        spec=dict(policies=policies,candidate=candidate,workload=workload)
        (args.output/'screen_summary.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.output/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
        names=list(policies)
        order=[(0,n) for n in names]+[(1,n) for n in reversed(names)]
        env=dict(os.environ);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        env.update(PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        for repeat,name in order:
            case=f'r{repeat}_{name}';keep,every,mode=policies[name];idle(case);status('running',case=case)
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'benchmarks/compare_fsdp_training.py','--backend','official','--model',workload['model'],
                '--keep-full-layers',str(keep),'--checkpoint-every',str(every),'--checkpoint-mode',mode,
                '--memory-budget-gib','42','--batch','1','--seq','1024','--warmup','3','--steps','12',
                '--reference-check','--output',str(args.output/case)]
            with (args.output/f'{case}.log').open('x') as stream:
                code=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            with (args.output/'status.log').open('a') as stream:stream.write(f'{case} returncode={code}\n')
            if code:raise RuntimeError(f'Confirmation {case} failed with {code}; do not select around it')
        confirmed=summarize_confirmation(args.output,spec)
        (args.output/'result.json').write_text(json.dumps(confirmed,indent=2)+'\n')
        status('finished',candidate=candidate,
            candidate_over_two_percent_in_both_orders=confirmed['candidate_over_two_percent_in_both_orders'])
    except Exception as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
