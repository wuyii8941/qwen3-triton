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
from training.flashoptim_convergence_job import Blocks
from training.convergence_rules import Plateau
from training.fsdp_validation import evaluate_full
from training.fsdp_best_parameters import BestParameters,local_tensor


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,default=Path('training/fsdp_convergence_protocol_v1.json'))
    p.add_argument('--backend',choices=['official','official_keep8','official_keep24','custom','custom_avg'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--keep-full-layers',type=int)
    p.add_argument('--checkpoint-every',type=int,default=0)
    p.add_argument('--checkpoint-mode',choices=['full','selective'],default='full')
    p.add_argument('--preflight-steps',type=int,help='Safety-only short run; cannot claim convergence')
    p.add_argument('--tiny',action='store_true',help='Synthetic pipeline test, requires --preflight-steps')
    args=p.parse_args()
    from training.fsdp_convergence_policy import resolve_policy
    policy_kwargs=dict(keep_full_layers=args.keep_full_layers,
        checkpoint_every=args.checkpoint_every,checkpoint_mode=args.checkpoint_mode)
    resolve_policy(args.backend,**policy_kwargs)
    from training.fsdp_provenance import source_snapshot
    source_root=Path(__file__).resolve().parents[1]
    initial_sources=source_snapshot(source_root)
    if args.tiny and args.preflight_steps is None:p.error('Tiny requires explicit preflight cutoff')
    if args.preflight_steps is not None and args.preflight_steps<1:p.error('Positive preflight steps required')
    cfg=json.loads(args.protocol.read_text())
    if args.tiny:
        cfg['sequence_length']=16
        cfg['validation']['every_updates']=1
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
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
        if args.backend.startswith('official'):
            from training.official_fsdp import shard_official_qwen
            policy=resolve_policy(args.backend,**policy_kwargs,
                layer_count=len(model.model.layers),tiny=args.tiny)
            keep=policy['keep_full_layers']
            result['optimization']=policy
            result['keep_full_layers']=keep
            model=shard_official_qwen(model,keep_full_layers=keep,
                checkpoint_every=args.checkpoint_every,checkpoint_mode=args.checkpoint_mode)
        else:
            from training.block_fsdp import shard_qwen,BlockFSDP
            model=shard_qwen(model)
            for unit in model.modules():
                if isinstance(unit,BlockFSDP):unit.reduce_average=args.backend=='custom_avg'
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
            if args.backend.startswith('official') and keep:
                from training.official_fsdp import reshard_official_parameters
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


if __name__=='__main__':main()
