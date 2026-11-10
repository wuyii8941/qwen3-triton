"""Isolated torchrun worker: identical real-data FSDP training workload.

No exports/checkpoints: only small measurements under --output. Timing excludes
I/O/H2D, includes forward/backward/optimizer, ends with max-rank wall latency.
CUDA synchronization is only at complete-step boundaries, not each operator.
"""
import argparse
import json
import os
import time
import statistics
import contextlib
import hashlib
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoModelForCausalLM,Qwen3ForCausalLM,Qwen3Config


def main():
    from training.fsdp_provenance import source_snapshot
    project_root=Path(__file__).resolve().parents[1]
    initial_sources=source_snapshot(project_root)
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['custom','custom_avg','custom_prefetch','official','official_checkpoint',
        'official_keep8','official_swiglu','official_keep8_swiglu'],required=True)
    parser.add_argument('--model',default='/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
    parser.add_argument('--tiny',action='store_true')
    parser.add_argument('--exchange-reduce',choices=['torch','triton'])
    parser.add_argument('--exchange-scope',choices=['decoder','head','all'],default='all')
    parser.add_argument('--seq',type=int,default=512)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--data',type=Path,default=Path('artifacts/convergence_corpus_v1/train.bin'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',action='store_true',help='Extra unmeasured step; rank0 CPU/CUDA trace')
    parser.add_argument('--reference-check',action='store_true',help='Untimed initial logits parity against full HF BF16')
    parser.add_argument('--keep-full-layers',type=int,help='Explicit retention override for official backend only')
    parser.add_argument('--memory-budget-gib',type=float,help='CUDA allocator cap; excludes context/NCCL/external processes')
    parser.add_argument('--accumulation',type=int,default=1)
    parser.add_argument('--gradient-sync',choices=['each','defer'],default='each')
    parser.add_argument('--checkpoint-every',type=int,default=0)
    parser.add_argument('--checkpoint-mode',choices=['full','selective'],default='full')
    parser.add_argument('--loss-chunk-size',type=int,default=0,help='Opt-in chunked loss; GPU correctness gate required')
    parser.add_argument('--ce-only-chunk-size',type=int,default=0,help='Opt-in CE-only chunking; native head GEMM unchanged')
    args = parser.parse_args()
    if args.exchange_reduce and (args.backend!='official' or args.accumulation!=1 or
            args.loss_chunk_size or args.ce_only_chunk_size):
        parser.error('Exchange requires plain official FSDP, original BF16 gradients, accumulation=1')
    if not args.exchange_reduce and args.exchange_scope!='all':
        parser.error('Exchange scope requires exchange backend')
    exchange_files=[project_root/'benchmarks'/p for p in
        ('fsdp_bf16_exchange_adapter.py','bf16_exchange_reduce.py')]
    def exchange_sources():
        return {str(p.relative_to(project_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in exchange_files}
    initial_exchange_sources=exchange_sources()
    if args.ce_only_chunk_size<0 or (args.ce_only_chunk_size and
            (not args.backend.startswith('official') or args.loss_chunk_size)):
        parser.error('CE-only loss requires official backend, nonnegative size and no linear-loss adapter')
    if args.loss_chunk_size<0 or (args.loss_chunk_size and not args.backend.startswith('official')):
        parser.error('Chunked loss requires official backend and nonnegative chunk size')
    if min(args.seq,args.batch,args.steps,args.accumulation)<1 or args.warmup<1: parser.error('Invalid workload')
    if (args.accumulation>1 or args.gradient_sync!='each') and not args.backend.startswith('official'):
        parser.error('Accumulation experiments require official FSDP2')
    if args.checkpoint_every<0 or (args.checkpoint_every and not args.backend.startswith('official')):
        parser.error('Checkpoint interval requires official FSDP2 and nonnegative value')
    if args.checkpoint_every and args.backend=='official_checkpoint':
        parser.error('All-layer and interval checkpoint are separate policies')
    if args.checkpoint_mode=='selective' and not (args.checkpoint_every or args.backend=='official_checkpoint'):
        parser.error('Selective mode requires checkpointed layers')
    if args.keep_full_layers is not None and (args.backend!='official' or args.keep_full_layers<0):
        parser.error('Explicit retention requires official backend and nonnegative layer count')
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    allocator_backend=torch.cuda.get_allocator_backend()
    if args.memory_budget_gib is not None and allocator_backend!='native':
        raise ValueError('Budget-controlled experiments require the native allocator backend')
    if args.memory_budget_gib is not None:
        total=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        fraction=args.memory_budget_gib*2**30/total
        if not 0<fraction<1:parser.error('Memory budget must be positive and below device capacity')
        torch.cuda.set_per_process_memory_fraction(fraction)
    dist.init_process_group('nccl',timeout=timedelta(minutes=15),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    rank,world = dist.get_rank(),dist.get_world_size()
    args.output.mkdir(parents=True,exist_ok=True)
    report = dict(complete=False,backend=args.backend,world_size=world,seq=args.seq,
        batch=args.batch,warmup=args.warmup,steps=args.steps,records=[],model=args.model,
        precision='FP32 master/AdamW/reduce, BF16 compute',attention='forced Flash SDPA',
        communication_environment={key:os.environ.get(key) for key in
            ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE')},
        memory_budget_gib=args.memory_budget_gib,
        accumulation=args.accumulation,gradient_sync=args.gradient_sync,
        effective_global_batch=world*args.batch*args.accumulation,
        allocator=dict(backend=allocator_backend,environment={key:os.environ.get(key)
            for key in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')},
            settings=torch.cuda.memory._snapshot().get('allocator_settings',{})),
        cache_paths={key:os.environ.get(key) for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR')})
    operator_context=contextlib.ExitStack()
    try:
        torch.manual_seed(20261106)
        if args.tiny:
            config = Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=128,
                num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
                tie_word_embeddings=False,attention_dropout=0.)
            config._attn_implementation='sdpa'
            model = Qwen3ForCausalLM(config).to(torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
                torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True)
        parameter_count=sum(p.numel() for p in model.parameters())
        report['global_parameters']=parameter_count
        vocab=model.config.vocab_size
        if args.backend.endswith('_swiglu'):
            from training.hf_operators import training_operators
            operator_context.enter_context(training_operators(model,swiglu=True,hf_semantics='triton_backward'))
        keep_full_layers=min(8,len(model.model.layers)) if 'keep8' in args.backend else 0
        if args.keep_full_layers is not None:keep_full_layers=args.keep_full_layers
        report['optimization']=dict(keep_full_layers=keep_full_layers,
            exchange_reduce=args.exchange_reduce,exchange_scope=args.exchange_scope,
            loss_chunk_size=args.loss_chunk_size,
            ce_only_chunk_size=args.ce_only_chunk_size,
            swiglu='hf_rounded_triton_forward_backward' if args.backend.endswith('_swiglu') else 'native')
        if args.loss_chunk_size:
            from training.fsdp_loss import install_chunked_qwen_loss
            install_chunked_qwen_loss(model,chunk_size=args.loss_chunk_size)
        if args.ce_only_chunk_size:
            from training.ce_only_loss import install_ce_only_qwen_loss
            install_ce_only_qwen_loss(model,chunk_size=args.ce_only_chunk_size)
        from training.fsdp_activation import checkpoint_layer_indices
        report['optimization'].update(checkpoint_every=args.checkpoint_every,checkpoint_mode=args.checkpoint_mode,
            checkpointed_layers=list(checkpoint_layer_indices(len(model.model.layers),
                all_layers=args.backend=='official_checkpoint',every=args.checkpoint_every)))
        if args.backend.startswith('custom'):
            from training.block_fsdp import shard_qwen,BlockFSDP
            if args.backend=='custom_prefetch':
                from training.prefetch_fsdp import shard_qwen_prefetched
                model=shard_qwen_prefetched(model)
            else:model=shard_qwen(model)
            if args.backend=='custom_avg':
                for unit in model.modules():
                    if isinstance(unit,BlockFSDP): unit.reduce_average=True
        else:
            from training.official_fsdp import shard_official_qwen
            model=shard_official_qwen(model,checkpoint=args.backend=='official_checkpoint',
                keep_full_layers=keep_full_layers,checkpoint_every=args.checkpoint_every,
                checkpoint_mode=args.checkpoint_mode)
        exchange_callbacks=[]
        if args.exchange_reduce:
            from torch.distributed.fsdp import FSDPModule
            from benchmarks.fsdp_bf16_exchange_adapter import BF16ExchangeReduceScatter
            for name,unit in model.named_modules():
                selected=(args.exchange_scope=='all' or
                    (args.exchange_scope=='decoder' and name.startswith('model.layers.')) or
                    (args.exchange_scope=='head' and name=='lm_head'))
                if isinstance(unit,FSDPModule) and selected:
                    callback=BF16ExchangeReduceScatter(original_bf16=True,
                        accumulation=args.accumulation,backend=args.exchange_reduce,debug=False)
                    unit.set_custom_reduce_scatter(callback)
                    exchange_callbacks.append((name,callback))
            if not exchange_callbacks:raise ValueError('No matching FSDP exchange units')
        model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,
            weight_decay=.01,fused=True)
        # Same blocks/order in each backend; distinct data across ranks/steps.
        from training.fsdp_accumulation import step_sample_indices,gradient_sync,sync_this_microbatch
        if args.tiny:
            torch.manual_seed(90+rank)
            shape=(args.batch,args.seq) if args.accumulation==1 else (args.accumulation,args.batch,args.seq)
            batches=[torch.randint(vocab,shape,device='cuda')
                     for _ in range(args.warmup+args.steps)]
        else:
            from training.flashoptim_convergence_job import Blocks
            data=Blocks(args.data,args.seq,training=True,seed=20261106)
            batches=[]
            for i in range(args.warmup+args.steps):
                indices=step_sample_indices(i,rank,world,args.batch,args.accumulation)
                group=torch.stack([torch.stack([data[j] for j in part]) for part in indices]).cuda()
                batches.append(group[0] if args.accumulation==1 else group)
        if args.reference_check:
            if args.tiny:raise ValueError('Reference check requires pretrained model')
            model.eval()
            reference_ids=batches[0] if args.accumulation==1 else batches[0][0]
            with torch.no_grad():observed=model(reference_ids,use_cache=False).logits
            if args.backend.startswith('official'):
                # Do not charge a retained candidate for simultaneously
                # keeping its full weights and an unrelated full HF oracle.
                from training.official_fsdp import reshard_official_parameters
                reshard_official_parameters(model)
            error=[None]
            if rank==0:
                try:
                    reference=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
                        torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).cuda().eval()
                    with torch.no_grad():expected=reference(reference_ids,use_cache=False).logits
                    torch.testing.assert_close(observed,expected,atol=.03,rtol=.01)
                    report['initial_hf_parity']=dict(max_abs=(observed.float()-expected.float()).abs().max().item())
                    del reference,expected
                except Exception as exc:error[0]=str(exc)
            dist.broadcast_object_list(error,src=0)
            if error[0]:raise RuntimeError(f'Initial HF parity failed: {error[0]}')
            del observed
            model.train();torch.cuda.empty_cache()
        dist.barrier();torch.cuda.synchronize()
        startup=time.perf_counter()
        def forward_backward(group,profile=False):
            parts=(group,) if args.accumulation==1 else group.unbind(0)
            total=None
            for j,ids in enumerate(parts):
                enabled=sync_this_microbatch(j,args.accumulation,args.gradient_sync)
                # Both forward and backward must be inside this context.
                with gradient_sync(model,enabled):
                    with torch.profiler.record_function('comparison.forward') if profile else contextlib.nullcontext():
                        part=model(ids,labels=ids,use_cache=False).loss
                        # Fixed S, labels==ids, no ignored tokens in this worker:
                        # every rank/microbatch contributes the same token count.
                        if args.accumulation>1:part=part/args.accumulation
                    with torch.profiler.record_function('comparison.backward') if profile else contextlib.nullcontext():
                        part.backward()
                    total=part.detach() if total is None else total+part.detach()
            return total
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            from training.allocator_metrics import COUNTERS,counter_deltas
            for i,ids in enumerate(batches):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                dist.barrier();torch.cuda.synchronize()
                allocator_before=torch.cuda.memory_stats()
                tick=time.perf_counter()
                loss=forward_backward(ids)
                optimizer.step()
                torch.cuda.synchronize()
                duration=time.perf_counter()-tick
                allocator_delta=counter_deltas(allocator_before,torch.cuda.memory_stats())
                # Outside timing; no per-parameter host sync in measured step.
                metrics=torch.tensor([duration,float(loss.detach()),torch.cuda.max_memory_allocated()/2**30,
                    torch.cuda.max_memory_reserved()/2**30]+[
                        allocator_delta[k] if allocator_delta[k] is not None else -1 for k in COUNTERS],
                    device='cuda',dtype=torch.float64)
                durations=[torch.empty_like(metrics) for _ in range(world)]
                dist.all_gather(durations,metrics)
                table=torch.stack(durations).cpu()
                row=dict(step=i+1,warmup=i<args.warmup,max_rank_seconds=table[:,0].max().item(),
                    loss=table[:,1].mean().item(),peak_allocated_gib=table[:,2].max().item(),
                    peak_reserved_gib=table[:,3].max().item())
                row['allocator_activity_max_rank']={k:(int(table[:,4+j].max().item())
                    if (table[:,4+j]>=0).all() else None) for j,k in enumerate(COUNTERS)}
                if not torch.isfinite(table).all(): raise RuntimeError('Nonfinite training metrics')
                report['records'].append(row)
                if rank==0:
                    (args.output/'result.json').write_text(json.dumps(report,indent=2))
                    print(json.dumps(row),flush=True)
        steady=[r['max_rank_seconds'] for r in report['records'] if not r['warmup']]
        report.update(complete=True,median_step_seconds=statistics.median(steady),
            global_tokens_per_second=world*args.batch*args.accumulation*args.seq/statistics.median(steady),
            elapsed_with_warmup_seconds=time.perf_counter()-startup,
            scope='Isolated steady-state step; GPU-resident real data, no finite-gradient scan in timing; not convergence')
        input_digest=hashlib.sha256(torch.stack(batches).cpu().numpy().tobytes()).hexdigest()
        input_digests=[None]*world
        dist.all_gather_object(input_digests,input_digest)
        report['manifest']=dict(pytorch=torch.__version__,cuda=torch.version.cuda,
            exchange_sources=initial_exchange_sources,
            nccl=torch.cuda.nccl.version(),gpu=torch.cuda.get_device_name(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            input_sha256=input_digest,input_sha256_by_rank=input_digests,
            implementation_sources=initial_sources)
        if args.profile:
            dist.barrier();torch.cuda.synchronize()
            profiler=(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False)
                if rank==0 else contextlib.nullcontext())
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION),profiler:
                optimizer.zero_grad(set_to_none=True)
                forward_backward(batches[-1],profile=True)
                with torch.profiler.record_function('comparison.optimizer'): optimizer.step()
                torch.cuda.synchronize()
            if rank==0:
                profiler.export_chrome_trace(str(args.output/'trace.json'))
                trace=json.loads((args.output/'trace.json').read_text())
                kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and 'dur' in e]
                comm=sum(e['dur'] for e in kernels if 'nccl' in e.get('name','').lower())
                report['profile']=dict(kernel_work_ms=sum(e['dur'] for e in kernels)/1000,
                    nccl_kernel_work_ms=comm/1000,note='Summed kernel work can overlap; not wall-time percentages')
        callback_counts={name:callback.calls for name,callback in exchange_callbacks}
        counts_by_rank=[None]*world
        dist.all_gather_object(counts_by_rank,callback_counts)
        if args.exchange_reduce and any(not counts or any(v<=0 for v in counts.values())
                for counts in counts_by_rank):
            raise ValueError('Exchange callback did not execute on every selected unit/rank')
        report['exchange_callback_counts_by_rank']=counts_by_rank
        report['exchange_provenance_contract']='Original BF16 unit grads, no FP32 microbatch accumulation; debug scan disabled in timed steps'
        unchanged=torch.tensor(int(source_snapshot(project_root)==initial_sources and
            exchange_sources()==initial_exchange_sources),device='cuda')
        dist.all_reduce(unchanged,op=dist.ReduceOp.MIN)
        if not unchanged.item():
            raise RuntimeError('Implementation sources changed during measurement; rerun frozen sources')
        if rank==0:(args.output/'result.json').write_text(json.dumps(report,indent=2))
    finally:
        operator_context.close()
        dist.destroy_process_group()


if __name__=='__main__':main()
