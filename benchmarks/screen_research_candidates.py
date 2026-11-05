"""Independent initial screens, NOT full convergence or production guarantees."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from benchmarks.research_candidates import linear_loss,make_optimizer,model_loss,tensor_storage_bytes


def relative(a,b):
    x,y=a.detach().float(),b.detach().float()
    return float((x-y).norm()/y.norm().clamp_min(1e-12))


def measure(fn,iterations=8):
    for _ in range(2):fn()
    rows=[]
    for _ in range(iterations):
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();fn();torch.cuda.synchronize()
        rows.append(dict(ms=(time.perf_counter()-start)*1000,
                         peak_bytes=torch.cuda.max_memory_allocated(),
                         peak_delta_bytes=torch.cuda.max_memory_allocated()-base))
    return dict(median_ms=statistics.median(r['ms'] for r in rows),samples=rows)


def loss_screen(seq):
    e=torch.randn(seq,2048,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=(torch.randn(151936,2048,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
    labels=torch.randint(0,151936,(seq,),device='cuda');labels[0]=-100
    fe=e.detach().float().requires_grad_();fw=w.detach().float().requires_grad_()
    oracle=torch.nn.functional.cross_entropy(torch.nn.functional.linear(fe,fw),labels)
    oracle_value=float(oracle);ref=torch.autograd.grad(oracle,(fe,fw))
    results={}
    for variant in ('native','cce','liger'):
        try:
            loss=linear_loss(variant,e,w,labels);grad=torch.autograd.grad(loss,(e,w))
            errors=[relative(a,b) for a,b in zip(grad,ref)]
            finite=all(bool(torch.isfinite(g).all()) for g in grad)
            results[variant]=dict(loss=float(loss),loss_abs=abs(float(loss)-oracle_value),
                gradient_relative_l2=errors,passed=finite and max(errors)<=.02 and abs(float(loss)-oracle_value)<=.01)
            del loss,grad
        except Exception as exc:
            results[variant]=dict(passed=False,error=f'{type(exc).__name__}: {exc}')
    del fe,fw,oracle,ref
    timings=[]
    for round_id,order in enumerate((('native','cce','liger'),('liger','native','cce'),('cce','liger','native'))):
        for variant in order:
            if not results[variant]['passed']:continue
            def fn():return torch.autograd.grad(linear_loss(variant,e,w,labels),(e,w))
            timings.append(dict(round=round_id,variant=variant,**measure(fn)))
    return dict(seq=seq,vocab=151936,hidden=2048,dtype='bf16',oracle_loss=oracle_value,
                checks=results,timings=timings,scope='Linear+CE forward/backward only, no gradient filtering for CCE; no model convergence.')


def optimizer_screen():
    initial=[torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16)*.02,
             torch.randn(4096,2048,device='cuda',dtype=torch.bfloat16)*.02]
    gradients=[torch.randn_like(p)*.001 for p in initial]
    results={};reference=None
    for variant in ('native','flashoptim'):
        model=torch.nn.ParameterList([torch.nn.Parameter(p.clone()) for p in initial])
        opt=make_optimizer(model,variant,1e-5)
        def step():
            for p,g in zip(model,gradients):p.grad=g.clone()
            opt.step()
        torch.cuda.synchronize();start=time.perf_counter();step();torch.cuda.synchronize()
        first_ms=(time.perf_counter()-start)*1000
        for _ in range(3):step()
        final=[p.detach().clone() for p in model]
        checkpoint=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),checkpoint)
        checkpoint_size=checkpoint.tell();checkpoint.seek(0)
        saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
        restored=torch.nn.ParameterList([torch.nn.Parameter(p.clone()) for p in initial])
        restored.load_state_dict(saved['model']);other=make_optimizer(restored,variant,1e-5)
        other.load_state_dict(saved['optimizer'])
        step()
        for p,g in zip(restored,gradients):p.grad=g.clone()
        other.step()
        resume_error=max(float((p-q).abs().max()) for p,q in zip(model,restored))
        if reference is None:reference=[p.detach().clone() for p in final]
        update_error=sum(float((p-r).float().square().sum()) for p,r in zip(final,reference))**.5
        update_norm=sum(float((p-r).float().square().sum()) for p,r in zip(reference,initial))**.5
        results[variant]=dict(first_step_ms=first_ms,state_bytes=tensor_storage_bytes(opt.state_dict()),
            checkpoint_bytes=checkpoint_size,update_error_relative_to_native_update=update_error/max(update_norm,1e-30),
            resume_max_abs=resume_error,finite=all(bool(torch.isfinite(p).all()) for p in model),
            timing=measure(step,20))
        del model,opt,restored,other,saved,checkpoint,final
    return dict(results=results,scope='Synthetic ~12.6M parameters, BF16/no master; compression is approximate, not an equality gate or full-model speedup.')


def load_model():
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-1.7B',
        local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().train()
    model.config.use_cache=False
    return model


def parity(variant,seq):
    model=load_model()
    ids=np.memmap('artifacts/convergence_corpus_v1/validation.bin',dtype='<u4',mode='r')
    tokens=torch.from_numpy(ids[:seq].astype(np.int64)).unsqueeze(0).cuda()
    reference=None;rows=[]
    for candidate in ('native','native',variant):
        model.zero_grad(set_to_none=True)
        loss=model_loss(model,tokens,candidate);loss.backward()
        if reference is None:
            reference=(float(loss),{n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None})
        else:
            numerator=denominator=0.;finite=True;count=0
            for n,p in model.named_parameters():
                if p.grad is None:continue
                x=p.grad.detach().float().cpu();y=reference[1][n].float()
                finite=finite and bool(torch.isfinite(x).all());count+=1
                numerator+=float((x-y).square().sum());denominator+=float(y.square().sum())
            rel=(numerator/max(denominator,1e-30))**.5
            rows.append(dict(variant=candidate,loss_abs=abs(float(loss)-reference[0]),
                gradient_relative_l2=rel,gradient_tensors=count,finite=finite,
                passed=finite and rel<=.02 and abs(float(loss)-reference[0])<=.01))
        del loss
    return dict(variant=variant,seq=seq,checks=rows,passed=all(r['passed'] for r in rows),
                scope='Complete Qwen loss/parameter-gradient check, including native repeat control; no training performance on failure.')


def train_screen(variant,lr,seq,steps):
    start=time.perf_counter();model=load_model()
    compiled=variant.startswith('compile_')
    base_variant=variant.removeprefix('compile_')
    optimizer=make_optimizer(model,base_variant,lr)
    def loss_fn(tokens):return model_loss(model,tokens,base_variant)
    training_loss=torch.compile(loss_fn,fullgraph=True,dynamic=False) if compiled else loss_fn
    train=np.memmap('artifacts/convergence_corpus_v1/train.bin',dtype='<u4',mode='r')
    val=np.memmap('artifacts/convergence_corpus_v1/validation.bin',dtype='<u4',mode='r')
    n=len(train)//seq;stride=104729
    import math
    while math.gcd(stride,n)!=1:stride+=2
    def evaluate():
        model.eval();values=[]
        with torch.no_grad():
            for i in range(16):
                tokens=torch.from_numpy(val[i*seq:(i+1)*seq].astype(np.int64)).unsqueeze(0).cuda()
                values.append(float(model(tokens,labels=tokens,use_cache=False).loss))
        model.train();return sum(values)/len(values)
    initial=evaluate();target=initial-.15;curve=[];samples=[];crossing=None
    print(json.dumps(dict(event='initial',loss=initial,target=target,elapsed=time.perf_counter()-start)),flush=True)
    cold_step_ms=None
    for step in range(1,steps+1):
        index=((step-1)*stride+20261106)%n
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        tokens=torch.from_numpy(train[index*seq:(index+1)*seq].astype(np.int64)).unsqueeze(0).cuda()
        loss=training_loss(tokens);loss.backward();optimizer.step();torch.cuda.synchronize()
        elapsed_ms=(time.perf_counter()-tick)*1000
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        if cold_step_ms is None:cold_step_ms=elapsed_ms
        if step>4:samples.append(dict(step=step,step_ms=elapsed_ms,peak_bytes=torch.cuda.max_memory_allocated()))
        if step%8==0 or step==steps:
            value=evaluate();row=dict(event='validation',step=step,loss=value,elapsed=time.perf_counter()-start)
            curve.append(row);print(json.dumps(row),flush=True)
            if value<=target and crossing is None:crossing=row
    return dict(variant=variant,lr=lr,seq=seq,steps=steps,initial_validation=initial,target=target,
        validation=curve,first_crossing=crossing,cold_step_ms=cold_step_ms,
        median_step_ms=statistics.median(r['step_ms'] for r in samples),
        peak_allocated_bytes=max(r['peak_bytes'] for r in samples),
        optimizer_state_bytes=tensor_storage_bytes(optimizer.state_dict()),elapsed_seconds=time.perf_counter()-start,
        scope='64-step initial screen, 16 fixed validation blocks, no test-based selection; NOT convergence. First crossing includes load/JIT/eval. LR-search costs reported separately.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['loss','optimizer','parity','train'],required=True)
    parser.add_argument('--variant',choices=['native','flashoptim','triton','compile_native','compile_flashoptim','compile_triton','muon','normuon','cce','liger'],default='native')
    parser.add_argument('--seq',type=int,default=512);parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--lr',type=float,default=1e-5);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.steps<=4 or args.seq<=0 or args.lr<=0:raise ValueError('Invalid screen dimensions')
    torch.set_num_threads(4);torch.manual_seed(20261106)
    torch.backends.cuda.matmul.allow_tf32=False
    manifest=dict(torch=torch.__version__,triton=importlib.metadata.version('triton'),
        transformers=importlib.metadata.version('transformers'),gpu=torch.cuda.get_device_name(),
        packages={p:importlib.metadata.version(p) for p in ('flashoptim','cut-cross-entropy','liger-kernel')},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=20261106,
        precision='BF16 params/grads/moments; FlashOptim quantized moments, master_weight_bits=None',
        shared_gpu_warning='No exclusive lock; all comparisons use matched new software stack, not historic Torch2.5 timing.')
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        if args.mode=='loss':result=loss_screen(args.seq)
        elif args.mode=='optimizer':result=optimizer_screen()
        elif args.mode=='parity':result=parity(args.variant,args.seq)
        else:result=train_screen(args.variant,args.lr,args.seq,args.steps)
    with args.output.open('x') as out:json.dump(dict(manifest=manifest,result=result),out,indent=2)
    print(json.dumps(dict(event='complete',output=str(args.output))),flush=True)


if __name__=='__main__':main()
