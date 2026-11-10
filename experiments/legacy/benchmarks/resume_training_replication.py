"""Explicit recovery of v2; preserve interrupted evidence and flag timing invalid."""
import hashlib,json,os
from pathlib import Path
import subprocess,sys,time
import torch
from benchmarks.run_training_replication import run_stage

def main():
    root=Path('artifacts/training_replication_v2');plan=json.loads((root/'plan.json').read_text())
    for source in ('training/convergence_job.py','training/triton_adamw.py','training/convergence_rules.py'):
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=plan['source_sha256'][source]:
            raise RuntimeError('Frozen training source changed: '+source)
    name='s512_seed20261104_native';directory=root/name;protocol=root/f'{name}.protocol.json'
    saved=torch.load(directory/'last.pt',map_location='meta',weights_only=False)
    assert saved['protocol_hash']==hashlib.sha256(protocol.read_bytes()).hexdigest()
    step=saved['step'];del saved
    recovery=dict(epoch=time.time(),resumed_job=name,checkpoint_step=step,
        timing_invalid_jobs=[name],reason='Queue disappeared; cause unknown; preserve interrupted segment, rollback event stream to durable checkpoint.')
    with (root/'recovery.json').open('x') as out:json.dump(recovery,out,indent=2)
    events=directory/'events.jsonl';backup=directory/'events.interrupted_original.jsonl'
    events.rename(backup)
    with events.open('x') as out:
        for line in backup.read_text().splitlines():
            row=json.loads(line)
            if row['step']<=step:out.write(line+'\n')
    cache=(root/f'cache_{name}').resolve()
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(cache/'inductor');os.environ['TRITON_CACHE_DIR']=str(cache/'triton')
    run_stage(root,name,protocol,'native',('--resume',))
    base=json.loads(protocol.read_text())
    for job in plan['jobs']:
        if job['seq']!=2048:continue
        name=f's2048_seed{job["seed"]}_{job["backend"]}'
        if (root/name/'summary.json').exists():continue
        cfg=dict(base,seq=2048,seed=job['seed'],eval_every=64)
        path=root/f'{name}.protocol.json'
        if not path.exists():
            with path.open('x') as out:json.dump(cfg,out,indent=2)
        if (root/name).exists():raise RuntimeError('Unexpected existing output: '+name)
        cache=(root/f'cache_{name}').resolve()
        os.environ['TORCHINDUCTOR_CACHE_DIR']=str(cache/'inductor');os.environ['TRITON_CACHE_DIR']=str(cache/'triton')
        run_stage(root,name,path,job['backend'])
        subprocess.run([sys.executable,'-m','benchmarks.summarize_training_replication','--root',str(root)],check=True)
    print('RECOVERY_QUEUE_COMPLETE',flush=True)

if __name__=='__main__':main()
