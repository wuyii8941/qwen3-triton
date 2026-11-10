"""Real Qwen graph correctness gates, then independent full-step experiments."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from benchmarks.check_tied_embedding import patch
from benchmarks.research_candidates import make_optimizer
from benchmarks.screen_research_candidates import load_model
from training.graphed_forward_backward import GraphedForwardBackward


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',choices=['gate_native','gate_embedding','native','embedding','graph_native','graph_embedding'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seq',type=int,default=512)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--unrestricted',action='store_true',help='Performance confirmation without deterministic workspace constraint')
    args=parser.parse_args()
    root=Path('artifacts/graphed_training_unrestricted_v1' if args.unrestricted else ('artifacts/graphed_training_v1' if (args.batch,args.seq)==(1,512)
              else f'artifacts/graphed_training_b{args.batch}_s{args.seq}_v1'))
    if args.worker is None:
        root.mkdir(exist_ok=False)
        records=[]
        names=('gate_native','gate_embedding','native','graph_native','embedding','graph_embedding',
               'graph_embedding','embedding','graph_native','native') if args.seq==512 else (
               'gate_embedding','embedding','graph_embedding','graph_embedding','embedding')
        if args.unrestricted:
            previous=json.loads(Path('artifacts/graphed_training_v1/result.json').read_text())
            if not previous.get('complete') or not all(r.get('passed') for r in previous['records'] if r['name'].startswith('gate')):
                raise RuntimeError('Prior full-model graph gates required')
            names=tuple(name for name in names if not name.startswith('gate'))
        for name in names:
            output=root/f'{len(records)}_{name}.json'
            with (root/f'{len(records)}_{name}.log').open('w') as stream:
                env=dict(os.environ)
                if args.unrestricted:env.pop('CUBLAS_WORKSPACE_CONFIG',None)
                result=subprocess.run([sys.executable,'-m','benchmarks.check_graphed_qwen',
                    '--worker',name,'--output',str(output),'--seq',str(args.seq),'--batch',str(args.batch)],stdout=stream,stderr=subprocess.STDOUT,env=env)
            row=json.loads(output.read_text()) if output.exists() else dict(error='No worker result')
            row.update(name=name,returncode=result.returncode);records.append(row)
            (root/'result.json').write_text(json.dumps(dict(complete=False,records=records),indent=2))
            print('GRAPH',name,row.get('passed'),row.get('median_step_ms'),row.get('error'),flush=True)
            if result.returncode or row.get('passed') is False:
                # Stop rather than benchmark a failed graph or relax tolerances.
                return
        (root/'result.json').write_text(json.dumps(dict(complete=True,records=records),indent=2))
        return
    torch.set_num_threads(4);torch.manual_seed(20261105)
    report=dict(worker=args.worker,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                batch=args.batch,seq=args.seq,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                scope='Fixed shape; native Flash SDPA, FlashOptim unchanged; graph covers F/B only.')
    try:
        is_gate=args.worker.startswith('gate')
        torch.use_deterministic_algorithms(is_gate)
        model=load_model()
        if 'embedding' in args.worker: patch(model)
        data=np.memmap('artifacts/convergence_corpus_v1/validation.bin',dtype='<u4',mode='r')
        n=args.batch*args.seq
        batches=[torch.from_numpy(data[i*n:(i+1)*n].astype(np.int64)).reshape(args.batch,args.seq).cuda() for i in range(20)]
        fn=torch.compile(lambda x:model(x,labels=x,use_cache=False).loss,fullgraph=True,dynamic=False)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            if is_gate:
                references=[]
                for batch in batches[:2]:
                    model.zero_grad(set_to_none=True)
                    loss=fn(batch);loss.backward()
                    references.append((float(loss.detach()),{n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}))
                    del loss
                runner=GraphedForwardBackward(model,fn,batches[0])
                checks=[]
                for batch,(value,grads) in zip(batches,references):
                    loss=runner.backward(batch)
                    num=den=0.;worst=0.
                    for name,p in model.named_parameters():
                        a,b=p.grad.detach().cpu().float(),grads[name].float()
                        err,norm=float((a-b).square().sum()),float(b.square().sum())
                        num+=err;den+=norm;worst=max(worst,(err/max(norm,1e-30))**.5)
                    checks.append(dict(loss_abs=abs(float(loss.detach())-value),global_relative_l2=(num/max(den,1e-30))**.5,
                                       max_parameter_relative_l2=worst))
                report.update(checks=checks,passed=all(c['loss_abs']<=.01 and c['global_relative_l2']<=.02
                                                      and c['max_parameter_relative_l2']<=.05 for c in checks))
            else:
                optimizer=make_optimizer(model,'flashoptim',1e-5)
                start=time.perf_counter()
                runner=GraphedForwardBackward(model,fn,batches[0]) if args.worker.startswith('graph') else None
                torch.cuda.synchronize()
                report['graph_setup_seconds']=time.perf_counter()-start
                report['setup_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                times=[];peaks=[];reserved=[];losses=[]
                for batch in batches:
                    if runner is None: optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    if runner is None:
                        loss=fn(batch);loss.backward()
                    else: loss=runner.backward(batch)
                    optimizer.step()
                    torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)*1000)
                    peaks.append(torch.cuda.max_memory_allocated());reserved.append(torch.cuda.max_memory_reserved())
                    losses.append(float(loss.detach()))
                report.update(passed=all(np.isfinite(losses)),median_step_ms=statistics.median(times[4:]),
                              all_step_ms=times,losses=losses,peak_allocated_bytes=max(peaks[4:]),
                              peak_reserved_bytes=max(reserved[4:]),first_step_ms=times[0])
    except Exception as exc:
        report.update(passed=False,error=f'{type(exc).__name__}: {exc}')
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
