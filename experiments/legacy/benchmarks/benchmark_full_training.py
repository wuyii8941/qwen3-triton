"""One isolated process per variant; full pretrained Qwen3 training, no save.

BF16 parameters/gradients/Adam moments, NOT FP32 master-weight mixed precision.
Record this distinction rather than silently comparing different optimizers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import os
from contextlib import nullcontext
import torch
import transformers
import triton
from transformers import AutoModelForCausalLM
from torch.nn.attention import sdpa_kernel,SDPBackend
from training.chunked_loss import chunked_linear_cross_entropy
from training.hf_operators import training_operators


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['hf','compile','chunked','fused_loss','rms','swiglu','qk','operators','swiglu_hf','swiglu_hf_bwd'],default='hf')
    p.add_argument('--seq',type=int,default=512)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--checkpointing',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--timing',choices=['diagnostic','steady'],default='diagnostic')
    p.add_argument('--memory-budget-gib',type=float)
    p.add_argument('--optimizer',choices=['native','foreach','triton'],default='native')
    p.add_argument('--adamw-math',choices=['legacy','rounded'],default='legacy')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if min(a.seq,a.batch,a.accumulation,a.steps)<1:raise ValueError('Positive dimensions required')
    torch.set_num_threads(4)
    torch.manual_seed(20261102)
    if a.memory_budget_gib is not None:
        total=torch.cuda.get_device_properties(0).total_memory
        if not 0<a.memory_budget_gib*2**30<=total:raise ValueError('Invalid allocator budget')
        torch.cuda.set_per_process_memory_fraction(a.memory_budget_gib*2**30/total)
    path=Path('/data1/tzh/models/Qwen/Qwen3-1.7B')
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().train()
    model.config.use_cache=False
    if a.checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    corpus=json.loads(Path('artifacts/frequency_wikitext_data.json').read_text())
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    ids=tok(corpus['splits']['validation']['text'][:150000],add_special_tokens=False)['input_ids']
    needed=a.batch*a.seq*a.accumulation
    if len(ids)<needed:raise ValueError('Insufficient fixed input tokens')
    batches=torch.tensor(ids[:needed],device='cuda').reshape(a.accumulation,a.batch,a.seq)
    if a.optimizer=='triton':
        from training.triton_adamw import TritonAdamW
        optimizer=TritonAdamW(model.parameters(),lr=1e-5,math_mode=a.adamw_math)
    elif a.optimizer=='foreach':optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=True,fused=False)
    else:optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,fused=True)
    def loss_fn(tokens):
        if a.variant in ('chunked','fused_loss'):
            h=model.model(tokens,use_cache=False).last_hidden_state
            loss_impl=chunked_linear_cross_entropy
            if a.variant=='fused_loss':
                from training.fused_loss import fused_linear_cross_entropy
                loss_impl=fused_linear_cross_entropy
            return loss_impl(h[:,:-1].reshape(-1,h.shape[-1]),
                model.lm_head.weight,tokens[:,1:].reshape(-1),chunk_size=128)
        return model(tokens,labels=tokens,use_cache=False).loss
    fn=torch.compile(loss_fn) if a.variant=='compile' else loss_fn
    report={'manifest':dict(variant=a.variant,seq=a.seq,batch=a.batch,accumulation=a.accumulation,
        layers=model.config.num_hidden_layers,dtype='bf16 parameters/grads/Adam moments; no FP32 master',
        optimizer=f'{a.optimizer} AdamW lr=1e-5',checkpointing=a.checkpointing,torch=torch.__version__,
        adamw_math=a.adamw_math if a.optimizer=='triton' else 'native',
        transformers=transformers.__version__,triton=triton.__version__,gpu=torch.cuda.get_device_name(),
        seed=20261102,use_cache=False,attention='forced native Flash SDPA',
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        allocator_budget_gib=a.memory_budget_gib,
        config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
        data_sha256=corpus['splits']['validation']['sha256'],
        operator_sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ['training/chunked_loss.py','training/fused_loss.py','training/hf_operators.py',
             'ops/triton/linear_ce_forward.py','ops/triton/ce_gradient.py','training/triton_adamw.py']},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(batches.cpu().numpy().tobytes()).hexdigest()),'samples':[]}
    setup_start=time.perf_counter()
    operator_context=training_operators(model,rms=a.variant in ('rms','operators'),
        swiglu=a.variant in ('swiglu','operators','swiglu_hf','swiglu_hf_bwd'),qk=a.variant in ('qk','operators'),
        hf_semantics='triton_backward' if a.variant=='swiglu_hf_bwd' else a.variant=='swiglu_hf')
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION),operator_context:
        for step in range(a.warmup+a.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();forward=backward=0.
            losses=[];micro_events=[]
            for tokens in batches:
                events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
                events[0].record();loss=fn(tokens)/a.accumulation;events[1].record()
                loss.backward();events[2].record()
                micro_events.append(events)
                if a.timing=='diagnostic':events[2].synchronize()
                losses.append(loss.detach())
            e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            e0.record();optimizer.step();e1.record();e1.synchronize()
            elapsed=(time.perf_counter()-start)*1000
            for events in micro_events:
                forward+=events[0].elapsed_time(events[1]);backward+=events[1].elapsed_time(events[2])
            row=dict(step=step,warmup=step<a.warmup,loss=float(torch.stack(losses).sum()),forward_ms=forward,
                backward_ms=backward,optimizer_ms=e0.elapsed_time(e1),step_ms=elapsed,
                tokens_per_second=needed/(elapsed/1000),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_delta_bytes=torch.cuda.max_memory_allocated()-base)
            if not torch.isfinite(loss):raise AssertionError('Nonfinite loss')
            report['samples'].append(row)
            print(json.dumps(row),flush=True)
        if a.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
                optimizer.zero_grad(set_to_none=True)
                with torch.profiler.record_function('train_forward'):
                    loss=fn(batches[0])
                with torch.profiler.record_function('train_backward'):loss.backward()
                with torch.profiler.record_function('train_optimizer'):optimizer.step()
                torch.cuda.synchronize()
            report['profiler_table']=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40)
            prof.export_chrome_trace(str(a.output.with_suffix('.trace.json')))
    rows=[r for r in report['samples'] if not r['warmup']]
    report['medians']={k:statistics.median(r[k] for r in rows) for k in rows[0] if k not in ('step','warmup')}
    report['total_run_seconds']=time.perf_counter()-setup_start
    if a.variant=='compile':
        from torch._dynamo.utils import counters
        report['compile_counters']={str(k):{str(n):int(v) for n,v in val.items()}
                                    for k,val in counters.items()}
    report['scope']=f'{a.timing} timing; compile/warmup excluded; fixed batch, no data loading'
    with a.output.open('x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()
