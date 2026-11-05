"""Validate interrupted snapshot and resume remaining jobs in original environment."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import torch
from benchmarks.run_training_replication import run_stage

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='v2');a=parser.parse_args()
    root=Path('artifacts')/f'training_replication_{a.tag}'
    plan=json.loads((root/'plan.json').read_text())
    for source in ('training/convergence_job.py','training/triton_adamw.py','training/convergence_rules.py'):
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==plan['source_sha256'][source]
    name='s2048_seed20261102_triton';directory=root/name;protocol=root/f'{name}.protocol.json'
    checkpoint=directory/'last.pt'
    candidate=checkpoint if checkpoint.exists() else directory/'last.tmp'
    torch.set_num_threads(4)
    saved=torch.load(candidate,map_location='cpu',weights_only=False)
    assert saved['protocol_hash']==hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert saved['optimizer_name']=='triton' and saved['step']>0
    assert saved['optimizer']['state'] and saved['model']
    assert all(bool(torch.isfinite(t).all()) for t in saved['model'].values())
    step=saved['step'];del saved
    if candidate!=checkpoint:
        with candidate.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(candidate,checkpoint)
        fd=os.open(directory,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    recovery_path=root/'recovery.json';recovery=json.loads(recovery_path.read_text())
    if name not in recovery['timing_invalid_jobs']:recovery['timing_invalid_jobs'].append(name)
    recovery.setdefault('additional_recoveries',[]).append(dict(job=name,step=step,epoch=time.time(),
        validation='Full CPU load, finite model tensors, optimizer and protocol verified; promoted last.tmp if needed'))
    recovery_path.write_text(json.dumps(recovery,indent=2))
    print('VERIFIED_RECOVERY',name,step,flush=True)
    for job in plan['jobs']:
        if job['seq']!=2048:continue
        current=f's2048_seed{job["seed"]}_{job["backend"]}'
        if (root/current/'summary.json').exists():continue
        path=root/f'{current}.protocol.json'
        if not path.exists():
            with path.open('x') as out:json.dump(json.loads(protocol.read_text()),out,indent=2)
        cache=(root/f'cache_{current}').resolve()
        os.environ['TORCHINDUCTOR_CACHE_DIR']=str(cache/'inductor');os.environ['TRITON_CACHE_DIR']=str(cache/'triton')
        extra=('--resume',) if current==name else ()
        if not extra and (root/current).exists():raise RuntimeError('Unexpected existing output: '+current)
        run_stage(root,current,path,job['backend'],extra)
        subprocess.run([sys.executable,'-m','benchmarks.summarize_training_replication','--root',str(root)],check=True)
    print('ALL_REMAINING_JOBS_FINISHED',flush=True)

if __name__=='__main__':main()
