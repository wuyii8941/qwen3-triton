"""Run independent initial screens; failure of one candidate never hides others."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    root=Path('artifacts')/f'research_screen_{args.tag}';root.mkdir(exist_ok=False)
    plan=dict(steps=64,seq=512,validation_blocks=16,eval_every=8,target_initial_loss_drop=.15,
        lr_grids=dict(native=[3e-6,1e-5,3e-5],muon=[1e-4,3e-4,1e-3],normuon=[1e-4,3e-4,1e-3],flashoptim=[1e-5]),
        aux_adam_lr=1e-5,seed=20261106,flashoptim_master_weight_bits=None,cce_filter_eps=None,
        interpretation='Initial screen only; no convergence claim, no test-based LR tuning, no comparison to historical software timings.',
        visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ('benchmarks/research_candidates.py','benchmarks/screen_research_candidates.py',__file__,
             'artifacts/research_sources_v1/Muon/muon.py','artifacts/research_sources_v1/NorMuon/normuon.py')})
    (root/'plan.json').write_text(json.dumps(plan,indent=2));stages=[];started=time.perf_counter()
    def run(name,arguments):
        if shutil.disk_usage(root).free<80*1024**3:raise RuntimeError('Less than80GiB free; do not start another screen')
        output=root/f'{name}.json'
        command=[sys.executable,'-m','benchmarks.screen_research_candidates','--output',str(output),*arguments]
        print('START',name,flush=True);tick=time.perf_counter();observations=[]
        with (root/f'{name}.log').open('x') as log:
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:
                log.write(line);log.flush()
                try:
                    row=json.loads(line)
                    if row.get('event')=='validation':
                        observations.append(dict(process_elapsed_seconds=time.perf_counter()-tick,job_event=row))
                        print(name,'step',row['step'],'val',round(row['loss'],5),flush=True)
                except json.JSONDecodeError:pass
            code=child.wait()
        stage=dict(name=name,returncode=code,wall_seconds=time.perf_counter()-tick,observed_validations=observations)
        stages.append(stage)
        (root/f'{name}.clock.json').write_text(json.dumps(stage,indent=2))
        (root/'status.json').write_text(json.dumps(dict(stages=stages,elapsed=time.perf_counter()-started,complete=False),indent=2))
        print('END',name,code,round(stage['wall_seconds'],2),flush=True)
        return json.loads(output.read_text())['result'] if code==0 and output.exists() else None
    run('optimizer',['--mode','optimizer'])
    for seq in (512,2048):run(f'loss_s{seq}',['--mode','loss','--seq',str(seq)])
    eligible=[]
    for variant in ('cce','liger'):
        result=run(f'parity_{variant}',['--mode','parity','--variant',variant])
        if result and result['passed']:eligible.append(variant)
    jobs=[(variant,lr) for variant in ('native','flashoptim','muon','normuon') for lr in plan['lr_grids'][variant]]
    jobs += [(variant,1e-5) for variant in eligible]
    # Predeclared fixed randomized order, independent of outcomes/speed.
    import random
    random.Random(20261106).shuffle(jobs)
    (root/'train_jobs.json').write_text(json.dumps(jobs,indent=2))
    for variant,lr in jobs:
        run(f'train_{variant}_{lr:g}',['--mode','train','--variant',variant,'--lr',str(lr),'--steps','64'])
    report=dict(stages=stages,elapsed_seconds=time.perf_counter()-started,complete=True,
                fusion_candidates_eligible=eligible,interpretation=plan['interpretation'])
    (root/'status.json').write_text(json.dumps(report,indent=2))
    print('ALL_INITIAL_SCREENS_TERMINATED',str(root),flush=True)


if __name__=='__main__':main()
