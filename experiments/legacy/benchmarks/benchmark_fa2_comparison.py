"""Independent FA2 screen. Never changes the long-running training environment.

Operator: FP32 math oracle + random-upstream gradients, then randomized timing.
Model: identical Qwen weights, loss/all-parameter gradient parity, then three
paired randomized-order step trials with native fused AdamW. No checkpoints.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F


def error(actual, expected):
    x, y = actual.detach().float(), expected.detach().float()
    return dict(relative_l2=float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp_min(1e-12)),
                max_abs=float((x-y).abs().max()), finite=bool(torch.isfinite(x).all()))


def timed(fn, iterations):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    samples=[]
    for _ in range(iterations):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();fn();end.record();end.synchronize()
        samples.append(start.elapsed_time(end))
    return dict(median_ms=statistics.median(samples),samples_ms=samples,
                peak_delta_bytes=torch.cuda.max_memory_allocated()-base)


def operators(flash, seq, dtype, batch, iterations):
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    torch.manual_seed(20261103)
    q=torch.randn(batch,seq,16,128,device='cuda',dtype=dtype,requires_grad=True)
    k=torch.randn(batch,seq,8,128,device='cuda',dtype=dtype,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    upstream=torch.randn_like(q)
    def native():
        # Match HF 4.51's repeat_kv, including its GQA materialization cost.
        return F.scaled_dot_product_attention(q.transpose(1,2),
            repeat_kv(k.transpose(1,2),2),repeat_kv(v.transpose(1,2),2),
            dropout_p=0.,is_causal=True).transpose(1,2)
    def fa2():
        return flash(q,k,v,dropout_p=0.,causal=True)
    # Math oracle deliberately computes in FP32, independently of both kernels.
    fq,fk,fv=(x.detach().float().requires_grad_() for x in (q,k,v))
    scores=fq.transpose(1,2)@fk.transpose(1,2).repeat_interleave(2,dim=1).transpose(-1,-2)/(128**.5)
    mask=torch.ones(seq,seq,device='cuda',dtype=torch.bool).triu(1)
    oracle=(scores.masked_fill(mask,float('-inf')).softmax(-1)@
            fv.transpose(1,2).repeat_interleave(2,dim=1)).transpose(1,2)
    oracle_grads=torch.autograd.grad(oracle,(fq,fk,fv),upstream.float())
    checks={}
    for name,fn in [('sdpa_hf_gqa',native),('fa2_compact_gqa',fa2)]:
        output=fn();grads=torch.autograd.grad(output,(q,k,v),upstream)
        checks[name]=dict(output=error(output,oracle),
                          gradients=[error(a,b) for a,b in zip(grads,oracle_grads)])
        for result in [checks[name]['output'],*checks[name]['gradients']]:
            if not result['finite'] or result['relative_l2']>(.02 if dtype==torch.bfloat16 else .005):
                raise AssertionError(f'{name} numerical gate failed: {result}')
        del output,grads
    del scores,oracle,oracle_grads,fq,fk,fv,mask
    timings=[]
    for round_idx in range(3):
        order=[('sdpa_hf_gqa',native),('fa2_compact_gqa',fa2)]
        random.Random(31+round_idx).shuffle(order)
        for name,fn in order:
            def forward():
                with torch.no_grad():return fn()
            def backward():
                return torch.autograd.grad(fn(),(q,k,v),upstream)
            timings.append(dict(round=round_idx,backend=name,
                                forward=timed(forward,iterations),forward_backward=timed(backward,iterations)))
    return dict(batch=batch,seq=seq,head_dim=128,q_heads=16,kv_heads=8,dtype=str(dtype),
                checks=checks,timings=timings,
                scope='Includes HF repeat_kv for SDPA; speedup is not pure attention-kernel attribution.')


def model_trial(seq, iterations, parity_only=False):
    import numpy as np
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-1.7B',
        local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().train()
    model.config.use_cache=False
    data=np.memmap('artifacts/convergence_corpus_v1/validation.bin',dtype='<u4',mode='r')
    tokens=torch.from_numpy(data[:seq].astype(np.int64)).unsqueeze(0).cuda()
    initial={} if parity_only else {name:t.detach().cpu().clone() for name,t in model.state_dict().items()}
    def configure(backend):
        model.config._attn_implementation=backend
        if not all(layer.self_attn.config._attn_implementation==backend for layer in model.model.layers):
            raise AssertionError('HF attention dispatch not applied to all layers')
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION) if backend=='sdpa' else nullcontext()
    # No optimizer update in parity; compare exact same pretrained parameters.
    reference=None;checks={};comparisons=[]
    backends=('sdpa','sdpa','flash_attention_2') if parity_only else ('sdpa','flash_attention_2')
    for backend in backends:
        model.zero_grad(set_to_none=True)
        with configure(backend):
            loss=model(tokens,labels=tokens,use_cache=False).loss;loss.backward()
        grads={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
        if reference is None:
            reference=(float(loss),grads)
        else:
            numerator=denominator=0.;max_abs=0.;parameter_errors=[]
            if grads.keys()!=reference[1].keys():raise AssertionError('Parameter gradient coverage differs')
            for name,g in grads.items():
                x,y=g.float(),reference[1][name].float()
                if not torch.isfinite(x).all():raise AssertionError('Nonfinite parameter gradient')
                squared_error=float((x-y).square().sum());squared_norm=float(y.square().sum())
                numerator+=squared_error;denominator+=squared_norm
                parameter_errors.append(dict(name=name,squared_error=squared_error,
                    relative_l2=(squared_error/max(squared_norm,1e-30))**.5))
                max_abs=max(max_abs,float((x-y).abs().max()))
            checks=dict(backend=backend,loss_sdpa=reference[0],loss_candidate=float(loss),
                loss_abs_difference=abs(float(loss)-reference[0]),
                all_parameter_gradient_relative_l2=(numerator/max(denominator,1e-30))**.5,
                gradient_max_abs=max_abs,gradient_tensors=len(grads),
                largest_gradient_errors=sorted(parameter_errors,key=lambda r:r['squared_error'],reverse=True)[:8])
            checks['passed']=checks['loss_abs_difference']<=.01 and checks['all_parameter_gradient_relative_l2']<=.02
            comparisons.append(checks)
            if not checks['passed'] and not parity_only:
                raise AssertionError(f'Model parity failed: {checks}')
        del loss,grads
    del reference
    if parity_only:
        return dict(seq=seq,batch=1,dtype='bf16',checks=comparisons,
                    passed=all(c['passed'] for c in comparisons),
                    scope='No parameter updates or timing; repeat native backward control before FA2 comparison.')
    trials=[]
    for round_idx in range(3):
        order=['sdpa','flash_attention_2'];random.Random(47+round_idx).shuffle(order)
        for backend in order:
            model.load_state_dict(initial);model.zero_grad(set_to_none=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,fused=True)
            samples=[]
            with configure(backend):
                for step in range(3+iterations):
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    loss=model(tokens,labels=tokens,use_cache=False).loss
                    loss.backward();optimizer.step();torch.cuda.synchronize()
                    seconds=time.perf_counter()-start
                    if not torch.isfinite(loss):raise AssertionError('Nonfinite training loss')
                    if step>=3:samples.append(dict(step_ms=seconds*1000,loss=float(loss),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            trials.append(dict(round=round_idx,backend=backend,samples=samples,
                median_step_ms=statistics.median(r['step_ms'] for r in samples)))
            del optimizer,loss
    return dict(seq=seq,batch=1,dtype='bf16',layers=len(model.model.layers),checks=checks,trials=trials,
                input_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
                scope='Full forward/backward/native fused AdamW step; fixed GPU input; no data loading/eval/checkpoint; warmup excluded, not convergence.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['operator','model','model-parity'],required=True)
    parser.add_argument('--seq',type=int,default=512)
    parser.add_argument('--dtype',choices=['fp16','bf16'],default='bf16')
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.seq,args.batch,args.iterations)<1:raise ValueError('Positive dimensions required')
    if args.mode.startswith('model') and (args.batch!=1 or args.dtype!='bf16'):
        parser.error('Model screen currently supports batch=1, dtype=bf16 only')
    # Import only here: --help and CPU error-metric tests need no FA2 install.
    from flash_attn import flash_attn_func
    torch.set_num_threads(4);torch.manual_seed(20261103)
    manifest=dict(torch=torch.__version__,flash_attn=importlib.metadata.version('flash-attn'),
        transformers=importlib.metadata.version('transformers'),gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_gpu_caveat='No exclusive GPU lock; inspect concurrent GPU processes for every run.')
    # Backend selection outside timing: do not charge per-call Python context
    # setup to SDPA alone. FA2 does not consult PyTorch's SDPA selector.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        result=(operators(flash_attn_func,args.seq,torch.float16 if args.dtype=='fp16' else torch.bfloat16,
                          args.batch,args.iterations) if args.mode=='operator' else
                model_trial(args.seq,args.iterations,parity_only=args.mode=='model-parity'))
    with args.output.open('x') as out:json.dump(dict(manifest=manifest,result=result),out,indent=2)
    print(json.dumps(dict(output=str(args.output),mode=args.mode,seq=args.seq,completed=True)),flush=True)


if __name__=='__main__':main()
