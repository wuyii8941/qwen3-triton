"""Single-GPU resumable continual-pretraining job with real I/O accounting.

Plateau convergence is scoped to this fixed WikiText task, never original
Qwen pretraining. Safety deadline is explicitly a NON-converged termination.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModelForCausalLM
from torch.nn.attention import sdpa_kernel,SDPBackend
from training.convergence_rules import Plateau
from training.triton_adamw import TritonAdamW


class Blocks(Dataset):
    def __init__(self,path,seq,offset=0,training=False,seed=20261102):
        self.data=np.memmap(path,dtype='<u4',mode='r');self.seq=seq
        self.n=len(self.data)//seq;self.offset=offset;self.training=training;self.seed=seed
        if self.n<1:raise ValueError('Corpus shorter than one sequence')
        self.stride=104729
        while math.gcd(self.stride,self.n)!=1:self.stride+=2
    def __len__(self):return self.n
    def __getitem__(self,i):
        index=((i+self.offset)*self.stride+self.seed)%self.n if self.training else i
        return torch.from_numpy(self.data[index*self.seq:(index+1)*self.seq].astype(np.int64))


def durable_save(value,path):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    directory=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


def main():
    entry=time.perf_counter()
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--optimizer',choices=['triton','flashoptim'],required=True)
    p.add_argument('--execution',choices=['eager','compile','graph'],default='eager')
    from training.optimization_policy import MODES
    p.add_argument('--optimizations',choices=MODES,default='off',help='Opt-in research modes; default unchanged')
    p.add_argument('--embedding-gate',type=Path,help='Passing deterministic batch/seq/accumulation gradient-gate JSON')
    p.add_argument('--microbatch',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--memory-budget-gib',type=float,help='Optional allocator cap, excludes context/library allocations')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--interrupt-after',type=int,help='Preflight only: exit after a durable checkpoint, no test evaluation')
    p.add_argument('--calibrate-only',action='store_true',help='Repeat initial validation three times, no training/test')
    a=p.parse_args()
    if a.execution=='graph' and a.optimizer!='flashoptim':
        p.error('Graph research path is validated with --optimizer flashoptim only')
    if a.microbatch<1 or a.accumulation<1:p.error('Positive microbatch and accumulation required')
    if a.execution=='graph' and (a.microbatch!=1 or a.accumulation!=1):
        p.error('Graph currently validated for B1/accumulation1 only; use compile for batching experiments')
    protocol_bytes=a.protocol.read_bytes();cfg=json.loads(protocol_bytes)
    protocol_hash=hashlib.sha256(protocol_bytes).hexdigest()
    if a.output.exists() and not a.resume:raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    if a.memory_budget_gib is not None:
        total=torch.cuda.get_device_properties(0).total_memory
        if not 0<a.memory_budget_gib*2**30<total:p.error('Invalid allocator budget')
        torch.cuda.set_per_process_memory_fraction(a.memory_budget_gib*2**30/total)
    model=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().train()
    model.config.use_cache=False
    from training.optimization_policy import prepare
    optimizer,restore_optimizations=prepare(model,mode=a.optimizations,optimizer_name=a.optimizer,
        lr=cfg['lr'],seq=cfg['seq'],embedding_gate=a.embedding_gate,batch=a.microbatch,accumulation=a.accumulation)
    def loss_fn(batch):
        return model(batch,labels=batch,use_cache=False).loss
    # Compile real HF loss, including its autograd graph. No silent fallback:
    # fullgraph rejects graph breaks. Native fused optimizer stays identical.
    forward_loss=torch.compile(loss_fn,fullgraph=True,dynamic=False) if a.execution in ('compile','graph') else loss_fn
    graph_runner=None
    plateau=Plateau(**cfg['plateau']);step=0;previous_elapsed=0.
    if a.resume:
        saved=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False)
        if saved['protocol_hash']!=protocol_hash or saved['optimizer_name']!=a.optimizer:raise ValueError('Resume mismatch')
        if saved.get('execution','eager')!=a.execution:raise ValueError('Resume execution mismatch')
        if saved.get('optimizations','off')!=a.optimizations:raise ValueError('Resume optimization mode mismatch')
        if (saved.get('microbatch',1),saved.get('accumulation',1))!=(a.microbatch,a.accumulation):
            raise ValueError('Resume batch/accumulation mismatch')
        if saved.get('memory_budget_gib')!=a.memory_budget_gib:raise ValueError('Resume allocator budget mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        plateau=Plateau(**saved['plateau']);step=saved['step'];previous_elapsed=saved['elapsed_seconds']
        clock=a.output/'checkpoint_clock.json'
        if clock.exists():
            timing=json.loads(clock.read_text())
            if timing['step']==step:previous_elapsed=max(previous_elapsed,timing['elapsed_seconds'])
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng']);del saved
    corpus=Path(cfg['corpus']);seq=cfg['seq']
    effective_batch=a.microbatch*a.accumulation
    if not a.calibrate_only and len(Blocks(corpus/'train.bin',seq))<effective_batch:
        raise ValueError('Corpus smaller than effective batch')
    val=DataLoader(Blocks(corpus/'validation.bin',seq),batch_size=1,num_workers=2,pin_memory=True)
    def elapsed():return previous_elapsed+time.perf_counter()-entry
    def log(row):
        row.update(step=step,elapsed_seconds=elapsed(),optimizer=a.optimizer,optimizations=a.optimizations,
                   microbatch=a.microbatch,accumulation=a.accumulation,samples_seen=step*effective_batch)
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        print(json.dumps(row),flush=True)
    def evaluate(loader):
        model.eval();total=0.;count=0
        with torch.no_grad():
            for batch in loader:
                batch=batch.cuda(non_blocking=True)
                # Shared evaluation path for every backend. Transformers 4.51
                # tracing in eval constructs an explicit mask incompatible
                # with forced Flash SDPA. Do not silently fall back at runtime:
                # only TRAINING is compiled, evaluation is explicitly eager.
                loss=loss_fn(batch)
                total+=float(loss);count+=1
        model.train();return total/count
    def checkpoint(path):
        durable_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            plateau=asdict(plateau),step=step,elapsed_seconds=elapsed(),protocol_hash=protocol_hash,
            optimizer_name=a.optimizer,execution=a.execution,optimizations=a.optimizations,
            microbatch=a.microbatch,accumulation=a.accumulation,memory_budget_gib=a.memory_budget_gib,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),path)
        (a.output/'checkpoint_clock.json').write_text(json.dumps(dict(step=step,elapsed_seconds=elapsed())))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        if a.calibrate_only:
            values=[evaluate(val) for _ in range(3)]
            result=dict(validation_losses=values,noise_range=max(values)-min(values),elapsed_seconds=elapsed())
            (a.output/'calibration.json').write_text(json.dumps(result,indent=2))
            log(dict(event='calibration_complete',**result));return
        if step==0:log(dict(event='initial_validation',loss=evaluate(val)))
        converged=False;reason='safety_deadline'
        while elapsed()<cfg['max_seconds']:
            data=DataLoader(Blocks(corpus/'train.bin',seq,offset=step*effective_batch,training=True,seed=cfg['seed']),batch_size=a.microbatch,
                num_workers=2,pin_memory=True,drop_last=True)
            from training.batching import accumulation_groups
            for group in accumulation_groups(data,a.accumulation):
                if a.execution=='graph':
                    batch=group[0].cuda(non_blocking=True)
                    if graph_runner is None:
                        from training.graphed_forward_backward import GraphedForwardBackward
                        graph_runner=GraphedForwardBackward(model,forward_loss,batch)
                    # Never clear/replace the captured .grad buffers. Optimizer
                    # remains outside the graph, with a new step every replay.
                    loss=graph_runner.backward(batch)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    loss=None
                    for micro in group:
                        raw=forward_loss(micro.cuda(non_blocking=True))
                        part=raw if a.accumulation==1 else raw/a.accumulation
                        part.backward()
                        loss=part.detach() if loss is None else loss+part.detach()
                optimizer.step();step+=1
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                if step%cfg['log_every']==0:log(dict(event='train',loss=float(loss)))
                if step%cfg['eval_every']==0:
                    value=evaluate(val);improved,converged=plateau.update(value)
                    log(dict(event='validation',loss=value,plateau=asdict(plateau)))
                    start=time.perf_counter()
                    checkpoint(a.output/'last.pt')
                    if improved:
                        link=a.output/'best.link'
                        os.link(a.output/'last.pt',link);os.replace(link,a.output/'best.pt')
                    log(dict(event='checkpoint',seconds=time.perf_counter()-start))
                    if a.interrupt_after is not None and step>=a.interrupt_after:
                        log(dict(event='preflight_interrupted',status='not_converged'))
                        return
                    if converged:reason='validation_plateau';break
                if elapsed()>=cfg['max_seconds']:break
            if converged or elapsed()>=cfg['max_seconds']:break
        if step%cfg['eval_every']:checkpoint(a.output/'last.pt')
        # Test is touched only after terminal state, never used by plateau.
        test=DataLoader(Blocks(corpus/'test.bin',seq),batch_size=1,num_workers=2,pin_memory=True)
        last_test_loss=evaluate(test)
        best_path=a.output/'best.pt'
        if best_path.exists():
            best=torch.load(best_path,map_location='cpu',weights_only=False)
            model.load_state_dict(best['model']);best_step=best['step'];del best
            best_test_loss=evaluate(test)
        else:best_step=step;best_test_loss=last_test_loss
        summary=dict(status='converged' if converged else 'not_converged',reason=reason,step=step,
            elapsed_seconds=elapsed(),test_loss=best_test_loss,last_test_loss=last_test_loss,best_step=best_step,
            plateau=asdict(plateau),protocol_hash=protocol_hash,execution=a.execution,optimizations=a.optimizations,
            microbatch=a.microbatch,accumulation=a.accumulation,effective_batch=effective_batch,
            memory_budget_gib=a.memory_budget_gib)
        if a.execution in ('compile','graph'):
            from torch._dynamo.utils import counters
            summary['compile_counters']={str(k):{str(n):int(v) for n,v in val.items()}
                for k,val in counters.items() if val and all(isinstance(v,int) for v in val.values())}
        with (a.output/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
        log(dict(event='terminal',**summary))


if __name__=='__main__':main()
