"""Separate complete compiled training pair; preflights gate formal jobs."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def main():
    root=Path('artifacts/flashoptim_convergence_v1');root.mkdir(exist_ok=False)
    base=json.loads(Path('training/convergence_preflight_v1.json').read_text())
    base.update(eval_every=250,log_every=25,max_seconds=21600)
    plan=dict(protocol=base,order=['triton','flashoptim'],execution='compile',
        precision='BF16 params/grads; no master, no gradient release; FlashOptim quantized states',
        timing='spawn to exit, includes compilation/loading/eval/durable checkpoints/final test; preprocessing excluded',
        quality_target='triton minimum validation +0.002, first observed crossing; safety timeout is not convergence',
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ('training/flashoptim_convergence_job.py','benchmarks/research_candidates.py','training/triton_adamw.py')})
    (root/'plan.json').write_text(json.dumps(plan,indent=2))
    def run(name,optimizer,protocol,extra=()):
        if shutil.disk_usage(root).free<40*1024**3:raise RuntimeError('Disk below40GiB')
        cache=(root/f'cache_{name}').resolve()
        os.environ['TORCHINDUCTOR_CACHE_DIR']=str(cache/'inductor');os.environ['TRITON_CACHE_DIR']=str(cache/'triton')
        start=time.perf_counter();print('START',name,list(extra),flush=True)
        with (root/f'{name}.log').open('a') as log:
            code=subprocess.run([sys.executable,'-m','training.flashoptim_convergence_job',
                '--protocol',str(protocol),'--output',str(root/name),'--optimizer',optimizer,
                '--execution','compile',*extra],stdout=log,stderr=subprocess.STDOUT).returncode
        with (root/'process_clocks.jsonl').open('a') as out:
            out.write(json.dumps(dict(name=name,resume='--resume' in extra,returncode=code,wall_seconds=time.perf_counter()-start))+'\n')
        print('END',name,code,flush=True)
        if code:raise RuntimeError('Job failed: '+name)
    pre=dict(base,eval_every=2,log_every=1,max_seconds=3600)
    prepath=root/'preflight_protocol.json';prepath.write_text(json.dumps(pre,indent=2))
    losses={}
    for optimizer in plan['order']:
        name='preflight_'+optimizer
        run(name,optimizer,prepath,('--interrupt-after','2'))
        run(name,optimizer,prepath,('--resume','--interrupt-after','4'))
        rows=[json.loads(l) for l in (root/name/'events.jsonl').read_text().splitlines()]
        losses[optimizer]=next(r['loss'] for r in reversed(rows) if r['event']=='validation')
    if abs(losses['triton']-losses['flashoptim'])>.01:raise RuntimeError('Preflight validation mismatch')
    (root/'preflight_gate.json').write_text(json.dumps(losses,indent=2))
    protocol=root/'frozen_protocol.json';protocol.write_text(json.dumps(base,indent=2))
    for optimizer in plan['order']:run(optimizer,optimizer,protocol)
    summaries={o:json.loads((root/o/'summary.json').read_text()) for o in plan['order']}
    (root/'summaries.json').write_text(json.dumps(summaries,indent=2))
    print('ALL_JOBS_FINISHED',flush=True)

if __name__=='__main__':main()
