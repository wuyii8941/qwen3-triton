"""Official full Qwen3-8B training with an explicit complete Triton GPU path.

The new entry defaults to full runtime. native/v13/adam are labeled controls.
FSDP2/autograd host orchestration and CUDA driver/DMA remain host services;
no native GPU arithmetic/copy/fill/collective exemption exists in full mode.
Every formal run has16quality+3warmup+8timed+1real profiled updates=28.
Full-model qualification is an independent review, never a raw-run boolean.
"""
import argparse,copy,json,os,time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoModelForCausalLM
from training.data import Blocks
from training.evaluation import validation_batches
from training.fsdp.official import shard_official_qwen,reshard_official_parameters
from training.input_manifest import validate
from qwen3_triton.model import model_for_variant
from qwen3_triton.kernels.gemm_tuning import load_plan,identity as gemm_identity
from qwen3_triton.runtime.manager import TrainingRuntime,CAP_GIB
from qwen3_triton.runtime.adamw import TritonAdamW,local
from qwen3_triton.runtime.memory import zeros as runtime_zeros
from qwen3_triton.runtime.metrics import accumulate_validation
from train_qwen3_vocab_v12 import compare_first_gradients,sha

ROOT=Path(__file__).resolve().parents[1]


def profiler(enabled):
    return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if enabled else nullcontext()


@torch.no_grad()
def evaluate(model,dataset,runtime,*,phase,trace_path=None):
    # Data order, batch4, full coverage and token weighting match V13. Only
    # explicit data-path operations/collectives and tracing differ.
    was_training=model.training;model.eval();model.zero_grad(set_to_none=True)
    rank=dist.get_rank();world=dist.get_world_size()
    try:
        with runtime.validation_memory():
            with profiler(trace_path is not None) as trace:
                totals=runtime_zeros((2,),device='cuda',dtype=torch.float64)
                for ids,valid in validation_batches(dataset,rank,world,4):
                    labels=ids.clone();labels[~valid]=-100;ids=ids.cuda();labels=labels.cuda()
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):out=model(ids,labels=labels,use_cache=False)
                    valid_count=int(valid.sum())
                    if valid_count:
                        assert torch.isfinite(out.loss)
                        tokens=valid_count*(dataset.seq-1);accumulate_validation(totals,out.loss,tokens)
                    del out
                runtime.reduce(totals)
                expected=len(dataset)*(dataset.seq-1);assert int(totals[1])==expected
                loss=float(totals[0]/totals[1]);reshard_official_parameters(model);torch.cuda.synchronize()
            if trace_path is not None:trace.export_chrome_trace(str(trace_path))
        # V13 controls physically release peer scratch during validation;
        # full mode needs peer AG/AR buffers and keeps its reserve active.
        extra=0 if runtime.kind in ['v13','adam'] else runtime.extra_bytes
        runtime.record_peak(phase,external_bytes=extra,reset=True)
        return dict(loss=loss,tokens=expected,blocks=len(dataset),batch_per_rank=4)
    finally:
        reshard_official_parameters(model);model.train(was_training)


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',choices=['full','native','v13','adam'],default='full')
    p.add_argument('--variant',choices=['triton_optimized','hf'],default='triton_optimized')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/fsdp_8b.json')
    p.add_argument('--input-manifest',type=Path,required=True)
    p.add_argument('--reference-raw-dir',type=Path,action='append',default=[])
    p.add_argument('--gemm-plan',type=Path,default=ROOT/'results/qwen3_triton_v10_gemm_plan_001.json')
    p.add_argument('--quality-steps',type=int,default=16);p.add_argument('--warmup-steps',type=int,default=3)
    p.add_argument('--measure-steps',type=int,default=8)
    p.add_argument('--full-validation',action='store_true');p.add_argument('--trace',action='store_true')
    a=p.parse_args()
    if a.runtime=='full' and a.variant!='triton_optimized':p.error('Full GPU runtime requires the complete Triton model; HF is an explicit native control')
    if a.quality_steps<1 or a.warmup_steps<1 or a.measure_steps<2:p.error('Need quality>=1, warmup>=1, timed>=2')
    for key in ['output','input_root','protocol','input_manifest','gemm_plan']:setattr(a,key,getattr(a,key).resolve())
    a.reference_raw_dir=[x.resolve() for x in a.reference_raw_dir]
    if not a.output.is_relative_to(Path('/data1/tzh')):p.error('Output must stay under/data1/tzh')
    if not a.input_root.is_dir():p.error('Existing input root required')
    return a


def main():
    args=arguments();os.chdir(args.input_root)
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
    torch.manual_seed(20261114);torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.cuda.set_per_process_memory_fraction(CAP_GIB*2**30/torch.cuda.get_device_properties().total_memory)
    full=args.runtime=='full'
    if full:dist.init_process_group('gloo',timeout=timedelta(minutes=30))
    else:dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=torch.device('cuda',torch.cuda.current_device()))
    cpu_group=dist.group.WORLD if full else dist.new_group(backend='gloo',timeout=timedelta(minutes=30))
    try:
        if dist.get_world_size()!=4:raise ValueError('Four ranks required')
        if rank==0:args.output.mkdir(parents=True,exist_ok=False)
        dist.barrier(group=cpu_group)
        protocol=json.loads(args.protocol.read_text());checked=[None]
        if rank==0:
            try:checked[0]=dict(identity=validate(args.input_manifest,args.protocol,protocol))
            except Exception as exc:checked[0]=dict(error=repr(exc))
        dist.broadcast_object_list(checked,group=cpu_group)
        if 'error' in checked[0]:raise RuntimeError(checked[0]['error'])
        if protocol['model_repo']!='Qwen/Qwen3-8B':raise ValueError('Official Qwen3-8B required')
        if args.variant=='triton_optimized':load_plan(args.gemm_plan)
        hf=AutoModelForCausalLM.from_pretrained(protocol['model'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa')
        config=copy.deepcopy(hf.config)
        if config.num_hidden_layers!=36 or config.hidden_size!=4096 or config.tie_word_embeddings:raise ValueError('Official untied36layer8B layout required')
        official_numel=sum(p.numel() for p in hf.parameters());assert official_numel==8190735360
        is_triton=args.variant!='hf'
        if is_triton:
            with torch.device('meta'):model=model_for_variant(config,args.variant)
            weights=hf.state_dict();model.load_hf_state_dict(weights,assign=True)
            assert set(dict(model.named_parameters()))==set(dict(hf.named_parameters()));del weights,hf
        else:model=hf
        setup_capture=full and args.trace and rank==0
        with profiler(setup_capture) as setup_trace:
            runtime=TrainingRuntime(args.runtime,cpu_group)
            with runtime.scope():
                if is_triton:
                    from vocab_model_v12 import shard_vocab_loss_model
                    model=shard_vocab_loss_model(model,keep_full_layers=36,checkpoint_every=1,skip_embedding_backward_gather=True,fused_vocab_dw=True).train()
                else:
                    model=shard_official_qwen(model,keep_full_layers=36,checkpoint_every=1,checkpoint_mode='full').train()
                    model.model.embed_tokens.set_unshard_in_backward(False);model.lm_head.set_reshard_after_forward(False)
                runtime.bind(model)
                parameters=dict(model.named_parameters())
                assert len(parameters)==399 and all(local(p).dtype==torch.float32 for p in parameters.values())
                groups=[dict(params=[p for name,p in parameters.items() if name!='lm_head.weight']),dict(params=[parameters['lm_head.weight']])]
                optimizer=(TritonAdamW(groups) if args.runtime in ['full','adam'] else torch.optim.AdamW(groups,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=True))
                torch.cuda.synchronize()
        setup_traces={}
        if setup_capture:
            path=args.output/'setup_trace.json.gz';setup_trace.export_chrome_trace(str(path));setup_traces[path.name]=sha(path)
        dist.barrier(group=cpu_group)  # host model construction/profile export complete
        with runtime.scope():
            runtime.record_peak('setup',reset=True)
            train=Blocks(Path(protocol['corpus'])/'train.bin',512,training=True,seed=protocol['seed'])
            validation=Blocks(Path(protocol['corpus'])/protocol['validation']['split'],512)
            rows=[];validation_rows=[];comparisons=[];traces=setup_traces;quality_warm_timed=args.quality_steps+args.warmup_steps+args.measure_steps
            if args.full_validation:
                value=evaluate(model,validation,runtime,phase='validation_0');validation_rows.append(dict(after_updates=0,**value))
                if rank==0:print(json.dumps(dict(event='full_validation',runtime=args.runtime,after_updates=0,**value)),flush=True)
                dist.barrier(group=cpu_group)
            total=quality_warm_timed+1
            for step in range(1,total+1):
                measured=args.quality_steps+args.warmup_steps<step<=quality_warm_timed
                profiled=step==total
                if step==args.quality_steps+1:torch.cuda.empty_cache()
                trace_name='initial_step_trace.json.gz' if step==1 else 'full_step_trace.json.gz'
                capture=args.trace and rank==0 and (step==1 or profiled)
                with profiler(capture) as trace:
                    ids_cpu=train[(step-1)*4+rank].unsqueeze(0).contiguous();ids=ids_cpu.cuda()
                    torch.cuda.synchronize();runtime.barrier();torch.cuda.synchronize();start=time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        out=model(ids,labels=ids,use_cache=False,**({'diagnostic_logits':step==1 and bool(args.reference_raw_dir)} if is_triton else {}))
                        out.loss.backward()
                    if step==1:
                        for reference in args.reference_raw_dir:
                            comparisons.append(compare_first_gradients(parameters,ids_cpu,out.logits,out.loss,reference,rank,checked[0]['identity'],strict_control=not is_triton))
                        # CPU archive review may vary substantially per rank.
                        # Coordinate host completion before another GPU signal
                        # barrier, so its timeout does not measure CPU file IO.
                        dist.barrier(group=cpu_group)
                    if not measured:
                        for name,p in parameters.items():
                            g=local(p.grad);assert g is not None and g.dtype==torch.float32 and torch.isfinite(g).all(),name
                        del g
                    optimizer.step()
                    loss=out.loss.detach().float().clone();runtime.reduce(loss);loss/=4;assert torch.isfinite(loss)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    # This instrumentation is in the trace; max-reducing a
                    # duration necessarily occurs after its measured interval.
                    elapsed_tensor=torch.tensor(elapsed,device='cpu',dtype=torch.float64).cuda();runtime.reduce(elapsed_tensor,'max');elapsed=float(elapsed_tensor)
                    global_loss=float(loss);del out,loss
                    torch.cuda.synchronize()
                if capture:
                    path=args.output/trace_name;trace.export_chrome_trace(str(path));traces[trace_name]=sha(path)
                if args.trace and (step==1 or profiled):dist.barrier(group=cpu_group)
                row=dict(step=step,global_loss=global_loss,measured=measured,profile_update=profiled,wall_seconds=elapsed,
                    includes_reference_review=step==1 and bool(args.reference_raw_dir))
                rows.append(row)
                if rank==0:print(json.dumps(dict(event='optimizer_update',runtime=args.runtime,variant=args.variant,**row)),flush=True)
                if step==quality_warm_timed:
                    training_peak=runtime.record_peak('timed_updates',reset=True)['torch']
                elif step==args.quality_steps:runtime.record_peak('quality_updates',reset=True)
                elif step==args.quality_steps+args.warmup_steps:runtime.record_peak('warmup_updates',reset=True)
                elif profiled:runtime.record_peak('profile_update',reset=True)
                if args.full_validation and step==args.quality_steps:
                    path=args.output/'full_validation_trace.json.gz' if full and args.trace and rank==0 else None
                    value=evaluate(model,validation,runtime,phase='validation_after_quality',trace_path=path);validation_rows.append(dict(after_updates=step,**value))
                    if path is not None:traces[path.name]=sha(path)
                    if rank==0:print(json.dumps(dict(event='full_validation',runtime=args.runtime,after_updates=step,**value)),flush=True)
                    dist.barrier(group=cpu_group)
            # The extra profiled step really updated parameters and moments.
            # Validate and report28, rather than hiding it as27+backward-only.
            for name,p in parameters.items():
                assert local(p).dtype==torch.float32 and torch.isfinite(local(p)).all(),name
                assert local(p.grad).dtype==torch.float32 and torch.isfinite(local(p.grad)).all(),name
                state=optimizer.state[p];assert float(state['step'])==total
                for key in ['exp_avg','exp_avg_sq']:assert local(state[key]).dtype==torch.float32 and torch.isfinite(local(state[key])).all(),(name,key)
            runtime.record_peak('post_update_checks')
            all_phase_peak={k:max(r['torch'][k] for r in runtime.phase_peaks) for k in ['allocated_gib','reserved_gib']}
            effective_all_phase_peak={k:max(r['effective'][k] for r in runtime.phase_peaks) for k in ['allocated_gib','reserved_gib']}
            reshard_official_parameters(model)
            report=runtime.report();extra=runtime.extra_bytes/2**30
            rank_result=dict(rank=rank,complete=True,parameter_count=399,optimizer_state_count=len(optimizer.state),updates=total,
                rows=rows,full_validation=validation_rows,gradient_comparisons=comparisons,all_master_grad_moment_finite_fp32=True,
                measured_training_peak=training_peak,effective_training_peak={k:v+extra for k,v in training_peak.items()},
                all_phase_peak=all_phase_peak,effective_all_phase_peak=effective_all_phase_peak,runtime=report,traces=traces,
                vocab_head_calls=dict(forward=model.lm_head.forward_calls,backward=model.lm_head.backward_calls) if is_triton else None)
        (args.output/f'rank{rank}.json').write_text(json.dumps(rank_result,indent=2)+'\n')
        reports=[None]*4;dist.all_gather_object(reports,rank_result,group=cpu_group)
        if rank==0:
            sources=[Path(__file__),ROOT/'scripts/train_qwen3_triton_8b.py',ROOT/'scripts/train_qwen3_vocab_v12.py',ROOT/'scripts/vocab_model_v12.py',args.protocol,args.input_manifest,
                ROOT/'scripts/research_python.sh',*sorted((ROOT/'qwen3_triton').rglob('*.py')),
                *sorted((ROOT/'training').rglob('*.py'))]
            if args.runtime in ['v13','adam']:sources.extend([ROOT/'scripts/fsdp_peer_rs_v13.py',ROOT/'scripts/symmetric_rs_v13.py'])
            if is_triton:sources.append(args.gemm_plan)
            formal=args.full_validation and args.trace and (args.quality_steps,args.warmup_steps,args.measure_steps)==(16,3,8) and bool(args.reference_raw_dir)
            result=dict(complete=True,full_runtime_qualified=False,performance_adopted=False,screen_only=not formal,
                experiment='full_GPU_runtime_v14',model='Qwen/Qwen3-8B',official_parameter_count=official_numel,layers=36,world_size=4,
                variant=args.variant,runtime=args.runtime,input_root=str(args.input_root),input_manifest_sha256=checked[0]['identity'],
                math_policy=dict(compute='BF16',master_gradient_moments='FP32',TF32=False,allow_bf16_reduced_precision_reduction=False),
                protocol=dict(quality_steps=args.quality_steps,warmup_steps=args.warmup_steps,measured_steps=args.measure_steps,
                    real_profile_updates=1,total_optimizer_updates=total,validation_points=[0,args.quality_steps] if args.full_validation else [],
                    validation_blocks=512 if args.full_validation else 0,step_timing='FWD+BWD+optimizer+loss statistics; max-rank elapsed instrumentation excluded from its own measured duration; extra profile update untimed.'),
                common_framework=dict(checkpoint_every=1,keep_full_layers=36,skip_embedding_backward_gather=True,
                    retain_lm_head=not is_triton,vocab_parallel=is_triton,fused_vocab_dw=is_triton,optimizer_groups='decoder+head_in_all_arms',
                    optimizer='TritonAdamW' if args.runtime in ['full','adam'] else 'native_fused_AdamW',batch_per_rank=1,sequence=512),
                gemm_plan=gemm_identity() if is_triton else None,ranks=reports,sources={str(p):sha(p) for p in sources},
                scope=__doc__,limitations='Raw run, not independent qualification. Full mode retains CPU FSDP/autograd/driver/DMA; all GPU kernels need phase-wide trace audit.16quality updates not convergence, fixed4A6000/B1S512, no inference claim.')
            (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if full:runtime.communication.close()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
