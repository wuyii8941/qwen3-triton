"""Sequential same-GPU preflight, frozen-protocol convergence jobs and report.

No fixed-step success criterion. A safety-limited job remains not_converged.
Process logs and incremental events are authoritative monitoring evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    root=Path('artifacts')/f'convergence_comparison_{a.tag}';root.mkdir(exist_ok=False)
    pipeline_start=time.perf_counter()
    def run(name,args):
        log=root/f'{name}.log'
        print('START',name,flush=True)
        with log.open('x') as f:
            result=subprocess.run([sys.executable,'-m','training.convergence_job',*args],stdout=f,stderr=subprocess.STDOUT)
        print('END',name,result.returncode,flush=True)
        if result.returncode:raise RuntimeError(f'{name} failed; inspect {log}')
    preflight='training/convergence_preflight_v1.json'
    # Each new comparison owns both save/resume preflights. The historical v1
    # run used an already-completed native save preflight outside its root;
    # future runs must not depend on or mutate that historical checkpoint.
    run('native_save_preflight',['--protocol',preflight,'--optimizer','native','--output',
        str(root/'native_preflight'),'--interrupt-after','2'])
    run('native_resume_preflight',['--protocol',preflight,'--optimizer','native','--output',
        str(root/'native_preflight'),'--resume','--interrupt-after','4'])
    run('triton_save_preflight',['--protocol',preflight,'--optimizer','triton','--output',
        str(root/'triton_preflight'),'--interrupt-after','2'])
    run('triton_resume_preflight',['--protocol',preflight,'--optimizer','triton','--output',
        str(root/'triton_preflight'),'--resume','--interrupt-after','4'])
    run('calibration',['--protocol',preflight,'--optimizer','native','--output',
        str(root/'calibration'),'--calibrate-only'])
    calibration=json.loads((root/'calibration/calibration.json').read_text())
    cfg=json.loads(Path(preflight).read_text())
    cfg.update(eval_every=250,log_every=25,max_seconds=21600)
    cfg['plateau']['min_delta']=max(.002,5*calibration['noise_range'])
    corpus=Path(cfg['corpus'])/'tokens_manifest.json'
    cfg['data_manifest_sha256']=hashlib.sha256(corpus.read_bytes()).hexdigest()
    cfg['decision_rule']='validation plateau, min10 checks, patience8; min_delta=max(.002,5*baseline noise range)'
    cfg['quality_target_rule']='native best validation + frozen min_delta; first crossing evaluated from both validation curves'
    cfg['timing_scope']='process startup, model/data loading, transfers, training, validation, fsync checkpoints and final evaluation; shared preprocessing/preflight excluded and separately reported'
    protocol=root/'frozen_protocol.json'
    with protocol.open('x') as f:json.dump(cfg,f,indent=2)
    print('PROTOCOL_FROZEN',protocol,flush=True)
    for optimizer in ('native','triton'):
        run(optimizer,['--protocol',str(protocol),'--optimizer',optimizer,'--output',str(root/optimizer)])
    summaries={name:json.loads((root/name/'summary.json').read_text()) for name in ('native','triton')}
    curves={}
    for name in summaries:
        curves[name]=[json.loads(line) for line in (root/name/'events.jsonl').read_text().splitlines()
                      if json.loads(line)['event'] in ('initial_validation','validation')]
    target=min(x['loss'] for x in curves['native'])+cfg['plateau']['min_delta']
    first_crossing={name:next((dict(step=x['step'],elapsed_seconds=x['elapsed_seconds'],loss=x['loss'])
                       for x in rows if x['loss']<=target),None) for name,rows in curves.items()}
    report=dict(summaries=summaries,quality_target=target,first_crossing=first_crossing,
        preprocessing_and_preflight_scope=cfg['timing_scope'],pipeline_elapsed_seconds=time.perf_counter()-pipeline_start,
        converged_both=all(x['status']=='converged' for x in summaries.values()),
        plateau_time_speedup=summaries['native']['elapsed_seconds']/summaries['triton']['elapsed_seconds'])
    if all(first_crossing.values()):
        report['matched_quality_speedup']=first_crossing['native']['elapsed_seconds']/first_crossing['triton']['elapsed_seconds']
    with (root/'comparison.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
