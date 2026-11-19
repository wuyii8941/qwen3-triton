"""Export the native Hugging Face reference for four-rank Qwen3-8B training."""
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import argparse,json,os,time
import torch
import torch.distributed as dist
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoModelForCausalLM
from qwen3_triton.training.data import Blocks
from qwen3_triton.training.evaluation import validation_batches
from qwen3_triton.training.fsdp.official import shard_official_qwen,reshard_official_parameters
from qwen3_triton.training.fsdp.activation_checkpoint import canonical_parameter_name
from qwen3_triton.training.input_manifest import validate
from qwen3_triton.gradient_archive import save as save_archive
from qwen3_triton.validation.training_reference import compare_first_gradients, local, sha


@torch.no_grad()
def evaluate(model,dataset):
    was_training=model.training;model.eval();model.zero_grad(set_to_none=True)
    rank=dist.get_rank();world=dist.get_world_size();nll=0.;count=0
    try:
        for ids,valid in validation_batches(dataset,rank,world,4):
            labels=ids.clone();labels[~valid]=-100;ids=ids.cuda();labels=labels.cuda()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out=model(ids,labels=labels,use_cache=False)
            valid_count=int(valid.sum())
            if valid_count:
                value=float(out.loss);assert torch.isfinite(out.loss)
                tokens=valid_count*(dataset.seq-1);nll+=value*tokens;count+=tokens
            del out
        totals=torch.tensor([nll,count],dtype=torch.float64,device='cuda');dist.all_reduce(totals)
        expected=len(dataset)*(dataset.seq-1);assert int(totals[1])==expected
        return dict(loss=float(totals[0]/totals[1]),tokens=expected,blocks=len(dataset),batch_per_rank=4)
    finally:
        reshard_official_parameters(model);model.train(was_training)
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('hf',),default='hf')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/fsdp_8b.json')
    parser.add_argument('--input-manifest',type=Path,required=True)
    parser.add_argument('--input-root',type=Path,help='Working directory for frozen relative input paths')
    parser.add_argument('--quality-steps',type=int,default=16)
    parser.add_argument('--warmup-steps',type=int,default=3)
    parser.add_argument('--measure-steps',type=int,default=8)
    parser.add_argument('--full-validation',action='store_true')
    parser.add_argument('--trace',action='store_true')
    parser.add_argument('--reference-raw-dir',type=Path,action='append',default=[])
    parser.add_argument('--save-reference',action='store_true')
    parser.add_argument('--keep-full-layers',type=int,choices=range(37),default=0)
    parser.add_argument('--checkpoint-every',type=int,default=1)
    parser.add_argument('--skip-embedding-backward-gather',action='store_true')
    parser.add_argument('--retain-lm-head',action='store_true')
    args=parser.parse_args()
    if args.checkpoint_every<0:parser.error('checkpoint-every must be nonnegative')
    if args.input_root:
        args.protocol=args.protocol.resolve();args.input_manifest=args.input_manifest.resolve()
        args.output=args.output.resolve()
        args.reference_raw_dir=[p.resolve() for p in args.reference_raw_dir]
        if not args.input_root.is_dir():parser.error('Input root must be an existing directory')
        os.chdir(args.input_root.resolve())
    if args.quality_steps<1 or args.warmup_steps<1 or args.measure_steps<2:parser.error('Need quality>=1, warmup>=1 and measured>=2 updates')
    rank=int(os.environ['RANK']);device=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(device);torch.set_num_threads(2)
    torch.manual_seed(20261114);torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.cuda.set_per_process_memory_fraction(41.875*2**30/torch.cuda.get_device_properties().total_memory)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=torch.device('cuda',device))
    try:
        assert dist.get_world_size()==4
        if rank==0:args.output.mkdir(parents=True,exist_ok=False)
        dist.barrier();protocol=json.loads(args.protocol.read_text());checked=[None]
        if rank==0:
            try:checked[0]=dict(identity=validate(args.input_manifest,args.protocol,protocol))
            except Exception as exc:checked[0]=dict(error=repr(exc))
        dist.broadcast_object_list(checked)
        if 'error' in checked[0]:raise RuntimeError(checked[0]['error'])
        assert protocol['model_repo']=='Qwen/Qwen3-8B'
        model=AutoModelForCausalLM.from_pretrained(protocol['model'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa')
        config=model.config;assert config.num_hidden_layers==36 and config.hidden_size==4096 and not config.tie_word_embeddings
        train=Blocks(Path(protocol['corpus'])/'train.bin',512,training=True,seed=protocol['seed'])
        validation=Blocks(Path(protocol['corpus'])/protocol['validation']['split'],512)
        model=shard_official_qwen(model,keep_full_layers=args.keep_full_layers,checkpoint_every=args.checkpoint_every,checkpoint_mode='full').train()
        if args.skip_embedding_backward_gather:
            model.model.embed_tokens.set_unshard_in_backward(False)
        if args.retain_lm_head:
            model.lm_head.set_reshard_after_forward(False)
        parameters=dict(model.named_parameters());assert len(parameters)==399 and all(local(p).dtype==torch.float32 for p in parameters.values())
        # Native fused AdamW cannot mix local Tensor and DTensor parameters in
        # one group. Use the same decoder/head grouping for every comparison arm.
        groups=[dict(params=[p for n,p in parameters.items() if n!='lm_head.weight']),
                dict(params=[parameters['lm_head.weight']])]
        optimizer=torch.optim.AdamW(groups,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=True)
        validation_rows=[];rows=[];comparisons=[];raw_first=None;training_peak=None
        if args.full_validation:
            v=evaluate(model,validation);validation_rows.append(dict(after_updates=0,**v))
            if rank==0:print(json.dumps(dict(event='full_validation',variant=args.variant,after_updates=0,**v)),flush=True)
        total=args.quality_steps+args.warmup_steps+args.measure_steps
        for step in range(1,total+1):
            if step==args.quality_steps+1:torch.cuda.empty_cache()
            if step==args.quality_steps+args.warmup_steps+1:torch.cuda.reset_peak_memory_stats()
            ids_cpu=train[(step-1)*4+rank].unsqueeze(0).contiguous();ids=ids_cpu.cuda()
            measured=step>args.quality_steps+args.warmup_steps
            torch.cuda.synchronize();dist.barrier();start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out=model(ids,labels=ids,use_cache=False)
                out.loss.backward()
            # Exact gradient comparisons are outside measured intervals and only
            # on update1, before either parameters or AdamW moments change.
            if step==1:
                for reference in args.reference_raw_dir:
                    comparisons.append(compare_first_gradients(parameters,ids_cpu,out.logits,out.loss,reference,rank,checked[0]['identity'],strict_control=True))
                if args.save_reference:
                    raw_first=dict(ids=ids_cpu,gradients={canonical_parameter_name(name):local(param.grad).detach().cpu() for name,param in parameters.items()},logits=out.logits.detach().cpu(),local_loss=out.loss.detach().cpu())
                    assert len(raw_first['gradients'])==399

            if not measured:
                for name,param in parameters.items():
                    grad=local(param.grad);assert grad is not None and grad.dtype==torch.float32 and torch.isfinite(grad).all(),name
                del grad
            optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if measured:
                elapsed_tensor=torch.tensor(elapsed,device='cuda',dtype=torch.float64);dist.all_reduce(elapsed_tensor,op=dist.ReduceOp.MAX);elapsed=float(elapsed_tensor)
            loss=out.loss.detach().float().clone();dist.all_reduce(loss);loss/=4;assert torch.isfinite(loss)
            rows.append(dict(step=step,global_loss=float(loss),measured=measured,wall_seconds=elapsed,includes_reference_review=step==1 and bool(args.reference_raw_dir or args.save_reference)))
            if rank==0:print(json.dumps(dict(event='optimizer_update',variant=args.variant,**rows[-1])),flush=True)
            del out
            if args.full_validation and step==args.quality_steps:
                v=evaluate(model,validation);validation_rows.append(dict(after_updates=step,**v))
                if rank==0:print(json.dumps(dict(event='full_validation',variant=args.variant,after_updates=step,**v)),flush=True)
        training_peak=dict(allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        for name,param in parameters.items():
            assert local(param).dtype==torch.float32 and torch.isfinite(local(param)).all(),name
            assert local(param.grad).dtype==torch.float32 and torch.isfinite(local(param.grad)).all(),name
            state=optimizer.state[param];assert float(state['step'])==total
            for key in ('exp_avg','exp_avg_sq'):assert local(state[key]).dtype==torch.float32 and torch.isfinite(local(state[key])).all(),(name,key)
        if args.trace:
            dist.barrier();optimizer.zero_grad(set_to_none=True)
            profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if rank==0 else nullcontext()
            with profile as trace:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out=model(ids,labels=ids,use_cache=False)
                    out.loss.backward()
                torch.cuda.synchronize()
            if rank==0:trace.export_chrome_trace(str(args.output/'training_trace.json'))
            del out
            dist.barrier()
        reshard_official_parameters(model)
        archive=save_archive(dict({'1':raw_first}),args.output/f'raw_rank{rank}.pt.gz',max_bytes=6*2**30) if args.save_reference else None
        rank_result=dict(vocab_head_calls=None,reference_archive=archive,gemm_plan=None,measured_training_peak=training_peak,rank=rank,complete=True,parameter_count=399,optimizer_state_count=len(optimizer.state),updates=total,rows=rows,full_validation=validation_rows,gradient_comparisons=comparisons,all_master_grad_moment_finite_fp32=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        (args.output/f'rank{rank}.json').write_text(json.dumps(rank_result,indent=2)+'\n')
        reports=[None]*4;dist.all_gather_object(reports,rank_result)
        if rank==0:
            files=[Path(__file__),ROOT/'scripts/train.py',args.protocol,args.input_manifest,*sorted((ROOT/'qwen3_triton').rglob('*.py'))]
            result=dict(complete=True,numerical_equivalence_qualified=False,performance_adopted=False,model='Qwen/Qwen3-8B',variant=args.variant,input_root=str(Path.cwd()),gemm_plan=None,world_size=4,layers=36,ranks=reports,input_manifest_sha256=checked[0]['identity'],math_policy=dict(compute='BF16',master_gradient_moments='FP32',TF32=False,allow_bf16_reduced_precision_reduction=False),experiment='hf_training_reference',screen_only=not(args.full_validation and args.quality_steps==16 and args.warmup_steps==3 and args.measure_steps==8),common_framework=dict(checkpoint_every=args.checkpoint_every,keep_full_layers=args.keep_full_layers,skip_embedding_backward_gather=args.skip_embedding_backward_gather,retain_lm_head=args.retain_lm_head,vocab_parallel=False,custom_collectives=False,custom_collective_schedule=False,head_gradient_preaveraged=False,fused_vocab_dw=False,vocab_dw_config=None,optimizer='native_fused_AdamW',optimizer_groups='decoder+head_in_all_arms',batch_per_rank=1,sequence=512),sources={str(p):sha(p) for p in files},scope=__doc__)
            (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
