"""Fixed effective-batch8 trajectory check of each path's selected microbatch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics

import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from benchmarks.training_dataflow_wave2 import loss_callable
from benchmarks.screen_research_candidates import load_model
from benchmarks.research_candidates import make_optimizer
from training.flashoptim_convergence_job import Blocks


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['baseline','consumed'])
    p.add_argument('--micro',type=int);p.add_argument('--output',required=True)
    p.add_argument('--matrix',default='artifacts/training_budget_expandable_wave2_v4/result.json')
    a=p.parse_args()
    if a.worker is None:
        matrix=json.loads(Path(a.matrix).read_text())
        if not matrix.get('complete'):raise RuntimeError('Complete budget matrix required')
        root=Path(a.output);root.mkdir(exist_ok=False)
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTORCH_ALLOC_CONF='expandable_segments:True')
        env.pop('CUBLAS_WORKSPACE_CONFIG',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        records=[]
        for v in ('baseline','consumed'):
            micro=matrix['selected'][v];out=root/f'{v}.json'
            with (root/f'{v}.log').open('w') as stream:
                result=subprocess.run([sys.executable,'-m','benchmarks.budget_trajectory_wave2',
                    '--worker',v,'--micro',str(micro),'--output',str(out)],env=env,stdout=stream,stderr=subprocess.STDOUT)
            row=json.loads(out.read_text()) if out.exists() else dict(error='No worker result')
            records.append(row)
            (root/'result.json').write_text(json.dumps(dict(complete=False,records=records),indent=2))
            print('BUDGET_TRAJECTORY',v,micro,result.returncode,row.get('error'),flush=True)
            if result.returncode or not row.get('complete'):return
        differences=[abs(x['loss']-y['loss']) for x,y in zip(records[0]['validation'],records[1]['validation'])]
        final=dict(complete=True,passed=max(differences)<=.01,max_validation_abs=max(differences),
            differences=differences,records=records,
            scope='128 updates/effective batch8, fixed data order, 16 validation blocks; not convergence or held-out test.')
        (root/'result.json').write_text(json.dumps(final,indent=2));print(json.dumps(final['differences']),flush=True)
        return
    torch.set_num_threads(4);torch.manual_seed(20261102)
    torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(0).total_memory)
    report=dict(variant=a.worker,microbatch=a.micro,accumulation=8//a.micro,effective_batch=8,
                budget_gib=16,steps=128,validation=[],training=[],step_ms=[])
    start=time.perf_counter()
    try:
        model=load_model();fn=loss_callable(model,a.worker);optimizer=make_optimizer(model,'flashoptim',1e-5)
        corpus=Path('artifacts/convergence_corpus_v1')
        train=Blocks(corpus/'train.bin',512,training=True,seed=20261102)
        val=Blocks(corpus/'validation.bin',512)
        def evaluate(step):
            model.eval()
            with torch.no_grad():
                values=[float(model(val[i].unsqueeze(0).cuda(),labels=val[i].unsqueeze(0).cuda(),use_cache=False).loss)
                        for i in range(16)]
            model.train();report['validation'].append(dict(step=step,loss=sum(values)/len(values),elapsed_seconds=time.perf_counter()-start))
            Path(a.output).write_text(json.dumps(report,indent=2))
            print('VALIDATION',a.worker,report['validation'][-1],flush=True)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            evaluate(0)
            for step in range(1,129):
                tick=time.perf_counter()
                ids=torch.stack([train[(step-1)*8+j] for j in range(8)]).cuda()
                optimizer.zero_grad(set_to_none=True);total=None
                for batch in ids.split(a.micro):
                    loss=fn(batch)/(8//a.micro);loss.backward()
                    total=loss.detach() if total is None else total+loss.detach()
                optimizer.step()
                if not torch.isfinite(total):raise RuntimeError('Non-finite loss')
                report['step_ms'].append((time.perf_counter()-tick)*1000)
                if step%16==0:report['training'].append(dict(step=step,loss=float(total)))
                if step%32==0:evaluate(step)
        report.update(complete=True,elapsed_seconds=time.perf_counter()-start,
            median_training_step_ms=statistics.median(report['step_ms'][4:]),
            scope='Real varied data + H2D + F/B/optimizer; validation excluded from step timing, included in elapsed')
    except Exception as exc:
        import traceback
        traceback.print_exc();report.update(complete=False,error=f'{type(exc).__name__}: {exc}')
    Path(a.output).write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
