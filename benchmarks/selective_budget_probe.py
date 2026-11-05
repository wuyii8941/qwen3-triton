"""One gated selective-recompute capacity candidate, then matched controls."""
import json
import os
from pathlib import Path
from benchmarks.training_dataflow_wave2 import child


def main():
    root=Path('artifacts/selective_budget_wave2_v1');root.mkdir(exist_ok=False);records=[]
    def run(name,cmd):
        r=child(root,name,cmd);records.append(r)
        (root/'result.json').write_text(json.dumps(dict(complete=False,records=records),indent=2))
        return r
    g=run('gate_selective',['gate','--variant','selective','--effective','8','--micro','8'])
    if not g.get('passed'):
        (root/'result.json').write_text(json.dumps(dict(complete=True,adopted=False,reason='gradient gate failed',records=records),indent=2));return
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    common=['--effective','8','--budget','16','--steps','12']
    probe=run('selective_mb8',['step','--variant','selective','--micro','8',*common])
    if probe.get('status')!='ok':
        (root/'result.json').write_text(json.dumps(dict(complete=True,adopted=False,reason=probe.get('status'),records=records),indent=2));return
    for variant,micro in (('baseline',2),('baseline',4),('consumed',4),('selective',8)):
        run(f'control_{variant}_mb{micro}',['step','--variant',variant,'--micro',str(micro),*common])
    (root/'result.json').write_text(json.dumps(dict(complete=True,records=records,
        scope='Pilot capacity/throughput, not a convergence or automatic adoption claim'),indent=2))


if __name__=='__main__':main()
