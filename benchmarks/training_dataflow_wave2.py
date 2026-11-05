"""Owned-gradient gate/ablation and fixed-effective-batch allocator sweep.

Independent child processes; no checkpoints, no convergence/test-set claims.
"""
import argparse
import hashlib
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
from benchmarks.screen_research_candidates import load_model
from benchmarks.research_candidates import make_optimizer
from training.tied_embedding import patch_tied_embedding
from training.owned_tied_loss import OwnedTiedLoss


def loss_callable(model,variant):
    if variant=='owned':return OwnedTiedLoss(model)
    if variant=='consumed':patch_tied_embedding(model,consume_proxy=True)
    if variant=='selective':
        from training.selective_checkpoint import enable_selective_checkpoint
        enable_selective_checkpoint(model)
    return torch.compile(lambda x:model(x,labels=x,use_cache=False).loss,fullgraph=True,dynamic=False)


def input_batch(batch,seq):
    data=np.memmap('artifacts/convergence_corpus_v1/validation.bin',dtype='<u4',mode='r')
    return torch.from_numpy(data[:batch*seq].astype(np.int64)).reshape(batch,seq).cuda()


def gate(a):
    torch.use_deterministic_algorithms(True)
    model=load_model();x=input_batch(a.effective,a.seq)
    reference=None;checks=[]
    for stage in ('baseline','repeat',a.variant):
        fn=loss_callable(model,'baseline' if stage=='repeat' else stage)
        model.zero_grad(set_to_none=True);value=0.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for batch in x.split(a.micro):
                loss=fn(batch)/(a.effective//a.micro);loss.backward();value+=float(loss.detach())
        grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        if reference is None:reference=(value,grads)
        else:
            if grads.keys()!=reference[1].keys():raise RuntimeError('Gradient keys mismatch')
            num=den=0.;rows=[]
            for name,g in grads.items():
                b=reference[1][name].float();g=g.float()
                err,norm=float((g-b).square().sum()),float(b.square().sum())
                num+=err;den+=norm;rows.append(dict(name=name,relative_l2=(err/max(norm,1e-30))**.5))
            whole=(num/max(den,1e-30))**.5;worst=max(r['relative_l2'] for r in rows)
            checks.append(dict(stage=stage,loss_abs=abs(value-reference[0]),global_relative_l2=whole,
                max_parameter_relative_l2=worst,passed=whole<=.02 and worst<=.05 and abs(value-reference[0])<=.01,
                worst_parameters=sorted(rows,key=lambda r:r['relative_l2'],reverse=True)[:5]))
        print('GATE',stage,checks[-1:],flush=True)
    return dict(checks=checks,passed=all(r['passed'] for r in checks),effective_batch=a.effective,microbatch=a.micro,
                seq=a.seq,variant=a.variant,batch=a.micro,accumulation=a.effective//a.micro,
                deterministic=True,consume_proxy=a.variant=='consumed',
                scope='Full real model deterministic loss/all-parameter gradient gate')


def step(a):
    a.phase='model_load';a.iteration=None
    if a.budget:
        total=torch.cuda.get_device_properties(0).total_memory
        if a.budget*2**30>=total:raise ValueError('Budget must be below physical device capacity')
        torch.cuda.set_per_process_memory_fraction(a.budget*2**30/total)
    model=load_model();fn=loss_callable(model,a.variant)
    optimizer=make_optimizer(model,'flashoptim',1e-5)
    x=input_batch(a.effective,a.seq);times=[];peaks=[];reserved=[];losses=[]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for i in range(a.steps):
            a.iteration=i
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();value=0.
            for batch in x.split(a.micro):
                a.phase='forward';loss=fn(batch)/(a.effective//a.micro)
                a.phase='backward';loss.backward()
                # Don't call .item or synchronize within the accumulation loop.
                value=value+loss.detach()
            a.phase='optimizer';optimizer.step();torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000);losses.append(float(value))
            peaks.append(torch.cuda.max_memory_allocated());reserved.append(torch.cuda.max_memory_reserved())
    median=statistics.median(times[4:])
    if a.budget and max(reserved)>a.budget*2**30:
        raise RuntimeError('Observed reserved memory exceeded requested allocator budget')
    return dict(status='ok',variant=a.variant,seq=a.seq,effective_batch=a.effective,microbatch=a.micro,
        accumulation=a.effective//a.micro,budget_gib=a.budget,median_step_ms=median,
        tokens_per_second=a.effective*a.seq*1000/median,all_step_ms=times,losses=losses,
        peak_allocated_bytes=max(peaks[4:]),peak_reserved_bytes=max(reserved[4:]),
        startup_peak_allocated_bytes=max(peaks),startup_peak_reserved_bytes=max(reserved),
        allocator_conf=os.environ.get('PYTORCH_ALLOC_CONF',os.environ.get('PYTORCH_CUDA_ALLOC_CONF')),
        first_step_ms=times[0],scope='Steady real HF+FlashOptim; cap is allocator-only, not total physical device usage')


def child(root,name,command):
    output=root/f'{name}.json';env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if command[0]=='gate':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    else:env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    start=time.time()
    with (root/f'{name}.log').open('w') as stream:
        result=subprocess.run([sys.executable,'-m','benchmarks.training_dataflow_wave2',*command,
            '--output',str(output)],stdout=stream,stderr=subprocess.STDOUT,env=env)
    row=json.loads(output.read_text()) if output.exists() else dict(status='error',error='No worker result')
    row.update(name=name,returncode=result.returncode,process_seconds=time.time()-start)
    print('DONE',name,row.get('passed',row.get('status')),row.get('median_step_ms'),flush=True)
    return row


def plan(a):
    root=Path(a.output);root.mkdir(exist_ok=False);records=[]
    def run(name,args):
        row=child(root,name,[*args,'--seq',str(a.seq)]);records.append(row)
        (root/'result.json').write_text(json.dumps(dict(complete=False,records=records),indent=2))
        if row.get('status')=='error':raise RuntimeError(f'Infrastructure/implementation error: {name}; stop, do not label OOM')
        return row
    if a.command=='owned':
        g=run('gate_owned',['gate','--variant','owned'])
        if not g.get('passed'):
            (root/'result.json').write_text(json.dumps(dict(complete=True,owned_adopted=False,records=records),indent=2));return
        for i,v in enumerate(('baseline','consumed','owned','owned','consumed','baseline')):
            run(f'r{i}_{v}',['step','--variant',v,'--steps','20'])
    else:
        accepted={}
        for micro in (1,2,4,8):
            old=Path(a.reuse_gates_from)/f'gate_consumed_mb{micro}.json' if a.reuse_gates_from else None
            g=json.loads(old.read_text()) if old is not None and old.exists() else None
            if (g is not None and g.get('passed') and g.get('effective_batch')==8
                    and g.get('microbatch')==micro and g.get('seq')==a.seq and g.get('variant')=='consumed'):
                # Normalize the legacy metadata schema without altering the
                # original result. These gates ran the same deterministic code.
                g.update(name=f'gate_consumed_mb{micro}',reused_from=str(old),deterministic=True,
                         reused_source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
                         batch=micro,accumulation=8//micro,consume_proxy=True)
                (root/f'gate_consumed_mb{micro}.json').write_text(json.dumps(g,indent=2))
                records.append(g)
                (root/'result.json').write_text(json.dumps(dict(complete=False,records=records),indent=2))
            else:
                g=run(f'gate_consumed_mb{micro}',['gate','--variant','consumed','--effective','8','--micro',str(micro)])
            accepted[micro]=g.get('passed',False)
        cases=[(v,b) for b in (1,2,4,8) for v in ('baseline','consumed') if v=='baseline' or accepted[b]]
        import random
        random.Random(20261107).shuffle(cases)
        for v,b in cases:
            run(f'{v}_mb{b}',['step','--variant',v,'--effective','8','--micro',str(b),
                '--budget',str(a.budget),'--steps','12'])
        for v in ('baseline','consumed'):
            good=[r for r in records if r.get('variant')==v and r.get('status')=='ok']
            if good:
                best=max(good,key=lambda r:r['tokens_per_second'])
                run(f'confirm_{v}_mb{best["microbatch"]}',['step','--variant',v,'--effective','8',
                    '--micro',str(best['microbatch']),'--budget',str(a.budget),'--steps','12'])
    (root/'result.json').write_text(json.dumps(dict(complete=True,records=records),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['gate','step','owned','budget'])
    p.add_argument('--variant',choices=['baseline','consumed','owned','selective'],default='baseline')
    p.add_argument('--effective',type=int,default=1);p.add_argument('--micro',type=int,default=1)
    p.add_argument('--seq',type=int,default=512);p.add_argument('--steps',type=int,default=12)
    p.add_argument('--budget',type=float,default=16);p.add_argument('--output',required=True)
    p.add_argument('--reuse-gates-from',help='Reuse completed matching wave2 gates; preserve original files')
    a=p.parse_args()
    if a.command in ('owned','budget'):return plan(a)
    torch.set_num_threads(4);torch.manual_seed(20261105)
    if a.effective%a.micro:raise ValueError('Microbatch must divide effective batch')
    start=time.time()
    try:
        if a.command=='gate':result=gate(a)
        else:
            # A budget is only imposed for effective-batch sweeps, not operator ablations.
            if a.effective==1:a.budget=0
            result=step(a)
    except torch.cuda.OutOfMemoryError as exc:
        result=dict(status='oom',error=str(exc),variant=a.variant,microbatch=a.micro,budget_gib=a.budget)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result=dict(status='error',passed=False,error=f'{type(exc).__name__}: {exc}')
    result.update(torch=torch.__version__,elapsed_seconds=time.time()-start)
    if result.get('status') in ('oom','error'):
        result.update(failed_phase=getattr(a,'phase',None),failed_iteration=getattr(a,'iteration',None))
    result['runtime_paths']={k:os.environ.get(k) for k in ('TMPDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH')}
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
