"""Protocol-driven real-data training; safety cutoff never means convergence."""
import argparse
import json
import os
import time
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM,Qwen3Config,Qwen3ForCausalLM
from training.data import Blocks
from training.stopping import Plateau
from training.evaluation import evaluate_full
from training.best_state import BestParameters,local_tensor
from training.policy import resolve_policy, validate_exchange


def run():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,default=Path('configs/fsdp_8b.json'))
    p.add_argument('--model',help='Local checkpoint override; no downloading')
    p.add_argument('--corpus',type=Path,help='Directory containing train/validation/test.bin')
    p.add_argument('--backend',choices=['official','custom'],default='official')
    p.add_argument('--exchange',choices=['native','torch_decoder'],default='native')
    p.add_argument('--memory-budget-gib',type=float,default=41.875)
    p.add_argument('--unchecked-exchange',action='store_true',help='Skip exact-BF16 checks; only after independent gates; matches historical debug=False experiment')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--keep-full-layers',type=int)
    p.add_argument('--checkpoint-every',type=int,help='Default: 1 for official, 0 for custom')
    p.add_argument('--checkpoint-mode',choices=['full'],default='full')
    p.add_argument('--preflight-steps',type=int,help='Safety-only short run; cannot claim convergence')
    p.add_argument('--tiny',action='store_true',help='Synthetic pipeline test, requires --preflight-steps')
    args=p.parse_args()
    if args.checkpoint_every is None:args.checkpoint_every=1 if args.backend=='official' else 0
    if args.keep_full_layers is None and args.backend=='official':
        args.keep_full_layers=32 if args.exchange=='torch_decoder' else 36
    policy_kwargs=dict(keep_full_layers=args.keep_full_layers,
        checkpoint_every=args.checkpoint_every,checkpoint_mode=args.checkpoint_mode)
    resolve_policy(args.backend,**policy_kwargs)
    from training.provenance import source_snapshot
    source_root=Path(__file__).resolve().parents[1]
    initial_sources=source_snapshot(source_root)
    if args.tiny and args.preflight_steps is None:p.error('Tiny requires explicit preflight cutoff')
    if args.preflight_steps is not None and args.preflight_steps<1:p.error('Positive preflight steps required')
    cfg=json.loads(args.protocol.read_text())
    if args.model:cfg['model']=args.model
    if args.corpus:cfg['corpus']=str(args.corpus.resolve())
    validate_exchange(args.backend,args.exchange,cfg)
    if args.unchecked_exchange and args.exchange=='native':p.error('Unchecked mode requires exchange')
    if not 0<args.memory_budget_gib<float('inf'):p.error('Positive finite memory budget required')
    if args.tiny:
        cfg['sequence_length']=16
        cfg['validation']['every_updates']=1
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    total=torch.cuda.get_device_properties().total_memory
    fraction=args.memory_budget_gib*2**30/total
    if not 0<fraction<1:p.error('Memory budget must be smaller than physical GPU memory')
    torch.cuda.set_per_process_memory_fraction(fraction)
    dist.init_process_group('nccl',timeout=timedelta(minutes=15),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    rank,world=dist.get_rank(),dist.get_world_size()
    error=[None]
    if rank==0:
        try:args.output.mkdir(parents=True,exist_ok=False)
        except Exception as exc:error[0]=str(exc)
    dist.broadcast_object_list(error,src=0)
    if error[0]:
        dist.destroy_process_group()
        raise RuntimeError(error[0])
    dist.barrier()
    entry=time.perf_counter()
    result=dict(complete=False,converged=False,backend=args.backend,protocol=cfg,validation=[],training=[],
        source_snapshot=initial_sources)
    def write():
        if rank==0:(args.output/'result.json').write_text(json.dumps(result,indent=2))
    try:
        if world!=cfg['world_size']:raise ValueError('Protocol world mismatch')
        torch.manual_seed(cfg['seed'])
        if args.tiny:
            config=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=128,
                num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                head_dim=16,tie_word_embeddings=False,attention_dropout=0.)
            config._attn_implementation='sdpa'
            model=Qwen3ForCausalLM(config).to(torch.bfloat16)
        else:
            model=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,
                torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True)
        if args.backend=='official':
            from training.fsdp.official import shard_official_qwen
            policy=resolve_policy(args.backend,**policy_kwargs,
                layer_count=len(model.model.layers),tiny=args.tiny)
            keep=policy['keep_full_layers']
            result['optimization']=policy
            result['keep_full_layers']=keep
            model=shard_official_qwen(model,keep_full_layers=keep,
                checkpoint_every=args.checkpoint_every,checkpoint_mode=args.checkpoint_mode)
        else:
            from training.fsdp.custom import shard_qwen,BlockFSDP
            model=shard_qwen(model)
            for unit in model.modules():
                if isinstance(unit,BlockFSDP):unit.reduce_average=True
        callbacks=[]
        if args.exchange=='torch_decoder':
            from training.fsdp.communication import BF16ExchangeReduceScatter
            for layer in model.model.layers:
                callback=BF16ExchangeReduceScatter(original_bf16=True,debug=not args.unchecked_exchange)
                layer.set_custom_reduce_scatter(callback)
                callbacks.append(callback)
        result['runtime']=dict(exchange=args.exchange,exchange_value_checks=not args.unchecked_exchange,
            memory_budget_gib=args.memory_budget_gib,allocator=torch.cuda.get_allocator_backend(),
            allocator_environment={k:os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')},
            current_layout=True,historical_performance_revalidated=False)
        model.train()
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),
            eps=cfg['epsilon'],weight_decay=cfg['weight_decay'],fused=True)
        seq,batch=cfg['sequence_length'],cfg['microbatch_per_rank']
        corpus=Path(cfg['corpus'])
        if args.tiny:
            class Synthetic:
                def __init__(self,n):
                    self.seq=seq;self.values=torch.randint(128,(n,seq))
                def __len__(self):return len(self.values)
                def __getitem__(self,i):return self.values[i%len(self.values)]
            train,validation=Synthetic(64),Synthetic(17)
            result['synthetic_preflight']=True
        else:
            train=Blocks(corpus/'train.bin',seq,training=True,seed=cfg['seed'])
            validation=Blocks(corpus/'validation.bin',seq)
        plateau=Plateau(**cfg['plateau']);best=BestParameters();best_step=0;snapshot_best=float('inf')
        def evaluate(step):
            nonlocal best_step,snapshot_best
            opt.zero_grad(set_to_none=True)
            row=evaluate_full(model,validation,cfg['validation']['batch_per_rank'])
            if args.backend=='official' and keep:
                from training.fsdp.official import reshard_official_parameters
                reshard_official_parameters(model)
                if any(p.dtype!=torch.float32 or not hasattr(p,'to_local') for p in model.parameters()):
                    raise RuntimeError('Best-state snapshot requires FP32 master shards')
            improved,done=plateau.update(row['loss'])
            # Stopping uses min_delta, but restore the actual lowest observed
            # validation loss, including improvements smaller than min_delta.
            if row['loss']<snapshot_best:
                # One FP32 CPU snapshot per rank; no repeated disk checkpoints.
                available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')))*1024
                needed=sum(local_tensor(p).numel()*p.element_size() for p in model.parameters())*world
                enough=torch.tensor(int(bool(best.values) or available>=needed+8*2**30),device='cuda')
                dist.all_reduce(enough,op=dist.ReduceOp.MIN)
                if not enough.item():
                    raise RuntimeError('Insufficient available CPU RAM for best parameters')
                best.save(model);best_step=step;snapshot_best=row['loss']
            row.update(step=step,elapsed_seconds=time.perf_counter()-entry)
            result['validation'].append(row);write()
            if rank==0:print(json.dumps(dict(event='validation',**row)),flush=True)
            return done
        evaluate(0)
        limit=args.preflight_steps if args.preflight_steps is not None else cfg['safety']['max_updates']
        reason='preflight_cutoff' if args.preflight_steps is not None else 'max_updates'
        for step in range(1,limit+1):
            expired=torch.tensor(int(time.perf_counter()-entry>=cfg['safety']['max_hours_per_backend']*3600),device='cuda')
            dist.all_reduce(expired,op=dist.ReduceOp.MAX)
            if expired.item():
                reason='safety_deadline';break
            torch.cuda.reset_peak_memory_stats()
            tick=time.perf_counter()
            ids=torch.stack([train[((step-1)*world+rank)*batch+j] for j in range(batch)]).cuda()
            opt.zero_grad(set_to_none=True)
            loss=model(ids,labels=ids,use_cache=False).loss
            loss.backward();opt.step();torch.cuda.synchronize()
            value=loss.detach().double();dist.all_reduce(value);value/=world
            if not torch.isfinite(value):raise RuntimeError('Nonfinite training loss')
            memory=torch.tensor([torch.cuda.max_memory_allocated()/2**30,
                torch.cuda.max_memory_reserved()/2**30],device='cuda',dtype=torch.float64)
            dist.all_reduce(memory,op=dist.ReduceOp.MAX)
            row=dict(step=step,loss=value.item(),seconds=time.perf_counter()-tick,
                max_rank_peak_allocated_gib=memory[0].item(),max_rank_peak_reserved_gib=memory[1].item())
            result['training'].append(row);write()
            if rank==0 and step%8==0:print(json.dumps(dict(event='training',**row)),flush=True)
            if step%cfg['validation']['every_updates']==0:
                if evaluate(step) and args.preflight_steps is None:
                    result['converged']=True;reason='validation_plateau';break
        best.restore(model)
        if result['converged']:
            result['test']=evaluate_full(model,Blocks(corpus/'test.bin',seq),cfg['validation']['batch_per_rank'])
        if source_snapshot(source_root)!=initial_sources:
            raise RuntimeError('Training source changed during convergence job')
        if callbacks and any(callback.calls!=len(result['training']) for callback in callbacks):
            raise RuntimeError('Communication callback coverage mismatch')
        result['callback_counts']=[callback.calls for callback in callbacks]
        result.update(complete=True,stop_reason=reason,best_step=best_step,
            best_validation=snapshot_best,plateau_reference_loss=plateau.best,elapsed_seconds=time.perf_counter()-entry,
            resume_checkpoint=False,best_parameters_restored=True)
        write()
    except Exception as error:
        result.update(complete=False,error=f'{type(error).__name__}: {error}')
        write()
        raise
    finally:
        dist.destroy_process_group()


def main():
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        run()


if __name__=='__main__':main()
