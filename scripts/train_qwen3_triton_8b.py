"""Selected V9 complete Qwen3-8B BF16 training/validation suite, no adoption claim.

The native HF arm must match the frozen first-gradient reference with a tighter
control gate and bitwise forward outputs. Candidate gradient gates stay unchanged.

Launch with torchrun on four idle A6000s. All arms share native FSDP2,
full-layer checkpointing and native fused AdamW. The output never labels a
completed run numerically qualified without a separate comparison/review.
"""
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import argparse,copy,hashlib,json,os,time
import torch
import torch.distributed as dist
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoModelForCausalLM
from training.data import Blocks
from training.evaluation import validation_batches
from training.fsdp.official import shard_official_qwen,reshard_official_parameters
from training.fsdp.activation_checkpoint import canonical_parameter_name
from training.input_manifest import validate
from qwen3_triton.audit import NoFallback
from qwen3_triton._model_optimized import VARIANTS,OptimizationOptions
from qwen3_triton.model import model_for_variant,PrecisionQwen3ForCausalLM as OptimizedTritonQwen3ForCausalLM
from qwen3_triton.gradient_archive import load as load_archive,save as save_archive
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*2**20),b''):digest.update(block)
    return digest.hexdigest()

def local(tensor):return tensor.to_local() if hasattr(tensor,'to_local') else tensor

def compare_first_gradients(parameters,ids,logits,loss,reference_dir,rank,identity,*,strict_control=False):
    """Every coordinate of all399localFP32gradients, outside performance timing."""
    result=json.loads((reference_dir/'result.json').read_text())
    assert result['input_manifest_sha256']==identity and result['math_policy']['allow_bf16_reduced_precision_reduction'] is False
    assert result['variant']=='hf' and result['layers']==36
    path=reference_dir/f'raw_rank{rank}.pt.gz'
    expected=result['ranks'][rank]['reference_archive'];assert sha(path)==expected['sha256']
    raw=load_archive(path)['1'];assert torch.equal(ids,raw['ids'])
    assert set(raw['gradients'])=={canonical_parameter_name(n) for n in parameters}
    rows=[];error2=reference2=0.
    for name,param in parameters.items():
        name=canonical_parameter_name(name);got=local(param.grad).detach().cpu();ref=raw['gradients'][name]
        assert got.shape==ref.shape and got.dtype==ref.dtype==torch.float32 and torch.isfinite(got).all() and torch.isfinite(ref).all()
        delta=float((got-ref).norm());norm=float(ref.norm());threshold=(.0001 if strict_control else .06)*max(norm,1e-8)+(1e-7 if strict_control else 1e-6)
        rows.append(dict(name=name,absolute_error_norm=delta,reference_norm=norm,relative_l2=delta/max(norm,1e-8),threshold=threshold,passed=delta<threshold))
        error2+=delta**2;reference2+=norm**2;del got
    assert len(rows)==399
    reference=raw['logits'].float();got=logits.detach().cpu().float()
    logit_error=float((got-reference).norm()/reference.norm().clamp_min(1e-8));loss_error=abs(float(loss.detach().cpu())-float(raw['local_loss']))
    passed=all(row['passed'] for row in rows)
    bitwise=torch.equal(logits.detach().cpu(),raw['logits']) and torch.equal(loss.detach().cpu(),raw['local_loss'])
    forward_passed=logit_error<.025 and loss_error<.01 and (bitwise or not strict_control)
    return dict(complete=True,rows=rows,passed=passed,global_relative_l2=(error2/max(reference2,1e-16))**.5,
        logits_relative_l2=logit_error,loss_abs_error=loss_error,forward_gate_passed=forward_passed,strict_native_control=strict_control,forward_bitwise_equal=bitwise,
        numerical_gate_passed=passed and forward_passed,reference_dir=str(reference_dir),
        reference_result_sha256=sha(reference_dir/'result.json'),
        scope='All-coordinate runtime comparison on firstupdate, same explicitBF16/FP32accum policyFalse. Dense candidategradients are not archived; original reference is losslessly archived.')
    
@torch.no_grad()
def evaluate(model,dataset,is_triton):
    was_training=model.training;model.eval();model.zero_grad(set_to_none=True)
    rank=dist.get_rank();world=dist.get_world_size();nll=0.;count=0
    try:
        for ids,valid in validation_batches(dataset,rank,world,4):
            labels=ids.clone();labels[~valid]=-100;ids=ids.cuda();labels=labels.cuda()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                with NoFallback() if is_triton else nullcontext():out=model(ids,labels=labels,use_cache=False)
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

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('hf',*VARIANTS,'fusion_only'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/fsdp_8b.json')
    parser.add_argument('--input-manifest',type=Path,default=ROOT/'artifacts/official_qwen3_inputs_v1.json')
    parser.add_argument('--quality-steps',type=int,default=16)
    parser.add_argument('--warmup-steps',type=int,default=3)
    parser.add_argument('--measure-steps',type=int,default=8)
    parser.add_argument('--full-validation',action='store_true')
    parser.add_argument('--trace',action='store_true')
    parser.add_argument('--reference-raw-dir',type=Path,action='append',default=[])
    parser.add_argument('--save-reference',action='store_true')
    parser.add_argument('--gemm-plan',type=Path,help='Explicit experimental SM86 plan; optimized variant only')
    args=parser.parse_args()
    if args.gemm_plan and args.variant!='triton_optimized':parser.error('A GEMM plan requires --variant triton_optimized')
    from qwen3_triton.kernels.gemm_tuning import load_plan,identity as gemm_plan_identity
    if args.gemm_plan:load_plan(args.gemm_plan)
    if args.save_reference and args.variant!='hf':parser.error('Only nativeHF reference exports dense gradients')
    if args.quality_steps<1 or args.warmup_steps<1 or args.measure_steps<2:parser.error('Need quality>=1, warmup>=1 and measured>=2 updates')
    rank=int(os.environ['RANK']);device=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(device);torch.set_num_threads(2)
    torch.manual_seed(20261114);torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False
    # Explicit FP32 accumulation in every arm; originalTrue-policy evidence stays separate.
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
        hf=AutoModelForCausalLM.from_pretrained(protocol['model'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa')
        config=copy.deepcopy(hf.config);assert config.num_hidden_layers==36 and config.hidden_size==4096 and not config.tie_word_embeddings
        is_triton=args.variant!='hf'
        if not is_triton:model=hf
        else:
            with torch.device('meta'):
                if args.variant=='fusion_only':
                    model=OptimizedTritonQwen3ForCausalLM(config,OptimizationOptions(linear_optimized=False,attention_optimized=False,loss_optimized=False))
                else:model=model_for_variant(config,args.variant)
            weights=hf.state_dict();model.load_hf_state_dict(weights,assign=True)
            assert set(dict(model.named_parameters()))==set(dict(hf.named_parameters()))
            assert all(param.data_ptr()==weights[name].data_ptr() for name,param in model.named_parameters())
            del weights
        del hf
        train=Blocks(Path(protocol['corpus'])/'train.bin',512,training=True,seed=protocol['seed'])
        validation=Blocks(Path(protocol['corpus'])/protocol['validation']['split'],512)
        model=shard_official_qwen(model,keep_full_layers=0,checkpoint_every=1,checkpoint_mode='full',vocab_parallel_head=False,vocab_parallel_embedding=False).train()
        parameters=dict(model.named_parameters());assert len(parameters)==399 and all(local(p).dtype==torch.float32 for p in parameters.values())
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=True)
        validation_rows=[];rows=[];comparisons=[];raw_first=None;training_peak=None
        if args.full_validation:
            v=evaluate(model,validation,is_triton);validation_rows.append(dict(after_updates=0,**v))
            if rank==0:print(json.dumps(dict(event='full_validation',variant=args.variant,after_updates=0,**v)),flush=True)
        total=args.quality_steps+args.warmup_steps+args.measure_steps
        for step in range(1,total+1):
            if step==args.quality_steps+1:torch.cuda.empty_cache()
            if step==args.quality_steps+args.warmup_steps+1:torch.cuda.reset_peak_memory_stats()
            ids_cpu=train[(step-1)*4+rank].unsqueeze(0).contiguous();ids=ids_cpu.cuda()
            measured=step>args.quality_steps+args.warmup_steps
            audit=is_triton and not measured
            torch.cuda.synchronize();dist.barrier();start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                with NoFallback() if audit else nullcontext():out=model(ids,labels=ids,use_cache=False)
                with NoFallback(backward=True) if audit else nullcontext():out.loss.backward()
            # Exact gradient comparisons are outside measured intervals and only
            # on update1, before either parameters or AdamW moments change.
            if step==1:
                for reference in args.reference_raw_dir:
                    comparisons.append(compare_first_gradients(parameters,ids_cpu,out.logits,out.loss,reference,rank,checked[0]['identity'],strict_control=args.variant=='hf'))
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
                v=evaluate(model,validation,is_triton);validation_rows.append(dict(after_updates=step,**v))
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
                    with NoFallback() if is_triton else nullcontext():out=model(ids,labels=ids,use_cache=False)
                    with NoFallback(backward=True) if is_triton else nullcontext():out.loss.backward()
                torch.cuda.synchronize()
            if rank==0:trace.export_chrome_trace(str(args.output/'training_trace.json'))
            del out
            dist.barrier()
        reshard_official_parameters(model)
        archive=save_archive(dict({'1':raw_first}),args.output/f'raw_rank{rank}.pt.gz',max_bytes=6*2**30) if args.save_reference else None
        rank_result=dict(reference_archive=archive,gemm_plan=gemm_plan_identity(),measured_training_peak=training_peak,rank=rank,complete=True,parameter_count=399,optimizer_state_count=len(optimizer.state),updates=total,rows=rows,full_validation=validation_rows,gradient_comparisons=comparisons,all_master_grad_moment_finite_fp32=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        (args.output/f'rank{rank}.json').write_text(json.dumps(rank_result,indent=2)+'\n')
        reports=[None]*4;dist.all_gather_object(reports,rank_result)
        if rank==0:
            files=[Path(__file__),args.protocol,args.input_manifest,ROOT/'training/fsdp/official.py',ROOT/'training/fsdp/activation_checkpoint.py',*sorted((ROOT/'qwen3_triton').rglob('*.py'))]
            if args.gemm_plan:files.append(args.gemm_plan)
            result=dict(complete=True,numerical_equivalence_qualified=False,performance_adopted=False,model='Qwen/Qwen3-8B',variant=args.variant,gemm_plan=gemm_plan_identity(),world_size=4,layers=36,ranks=reports,input_manifest_sha256=checked[0]['identity'],math_policy=dict(compute='BF16',master_gradient_moments='FP32',TF32=False,allow_bf16_reduced_precision_reduction=False),common_framework=dict(checkpoint_every=1,keep_full_layers=0,vocab_parallel=False,custom_collectives=False,optimizer='native_fused_AdamW',batch_per_rank=1,sequence=512),sources={str(p):sha(p) for p in files},scope=__doc__)
            (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
