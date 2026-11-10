"""Multi-rank Qwen3 block-sharding parity, tied and untied weights."""
import os
import copy
import json
import argparse
import tempfile
from types import SimpleNamespace
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM
from training.block_fsdp import shard_qwen, BlockFSDP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--exchange-reduce', choices=['torch','triton'],
        help='Experimental original-BF16 transport through FSDP2 custom comm interface')
    parser.add_argument('--deterministic', action='store_true',
        help='Deterministic numerical gate only; does not change benchmark execution')
    parser.add_argument('--checkpoint-test', action='store_true')
    parser.add_argument('--export-test', action='store_true')
    parser.add_argument('--backend',choices=['custom','official'],default='custom')
    parser.add_argument('--reduce-average',action='store_true')
    parser.add_argument('--prefetch',action='store_true',help='Experimental asynchronous prefetch gate')
    parser.add_argument('--keep-full-layers',type=int,default=0)
    parser.add_argument('--swiglu',action='store_true',help='HF-rounded Triton SwiGLU ablation')
    parser.add_argument('--head-dim',type=int,choices=[16,128],default=16)
    parser.add_argument('--accumulation',type=int,choices=[1,2],default=1)
    parser.add_argument('--gradient-sync',choices=['each','defer'],default='each')
    parser.add_argument('--checkpoint-every',type=int,default=0)
    parser.add_argument('--checkpoint-mode',choices=['full','selective'],default='full')
    parser.add_argument('--loss-chunk-size',type=int,default=0,help='Opt-in chunked loss gate; zero keeps HF loss')
    parser.add_argument('--ce-only-chunk-size',type=int,default=0,help='CE-only adapter; full native logits remain')
    parser.add_argument('--loss-backward-impl',choices=['default','native_ce','native_ce_padded'],default='default')
    parser.add_argument('--seq',type=int,default=11,help='Gate sequence length; use >chunk size to test real chunking')
    options = parser.parse_args()
    if options.exchange_reduce and (options.backend!='official' or not options.bf16 or
            options.accumulation!=1 or options.checkpoint_test or options.export_test):
        parser.error('Exchange gate requires official BF16, accumulation=1 and no export/checkpoint')
    if options.ce_only_chunk_size<0 or (options.ce_only_chunk_size and
            (options.backend!='official' or options.loss_chunk_size)):
        parser.error('CE-only adapter requires official backend, nonnegative size and no linear-loss adapter')
    if options.seq<2:parser.error('Gate sequence length must be >=2')
    if options.loss_backward_impl!='default' and not options.loss_chunk_size:
        parser.error('Loss backward variant requires a chunked loss gate')
    if options.loss_chunk_size<0 or (options.loss_chunk_size and
            (options.backend!='official' or options.accumulation!=1)):
        parser.error('Chunked loss gate requires official backend, accumulation=1, nonnegative size')
    if options.export_test and not options.bf16: parser.error('Export comparison requires --bf16')
    if (options.keep_full_layers or options.swiglu) and options.backend!='official':
        parser.error('New hybrid gates require official FSDP')
    if (options.accumulation>1 or options.gradient_sync!='each') and options.backend!='official':
        parser.error('Accumulation gate requires official FSDP2')
    if options.checkpoint_every<0 or (options.checkpoint_every and options.backend!='official'):
        parser.error('Layer checkpoint gate requires official backend and nonnegative interval')
    if options.checkpoint_mode=='selective' and not options.checkpoint_every:
        parser.error('Selective mode requires checkpointed layers')
    if options.prefetch and (options.backend!='custom' or options.reduce_average):
        parser.error('Prefetch is an isolated custom SUM ablation')
    if options.backend=='official' and (options.checkpoint_test or options.export_test or not options.bf16):
        parser.error('Official gate currently supports BF16 gradient/update parity only')
    dtype = torch.bfloat16 if options.bf16 else torch.float32
    torch.use_deterministic_algorithms(options.deterministic)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    try:
        for tied in (False, True):
            torch.manual_seed(8)
            hidden=4*options.head_dim
            config = Qwen3Config(vocab_size=97, hidden_size=hidden, intermediate_size=hidden*3//2,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=options.head_dim, tie_word_embeddings=tied, attention_dropout=0.)
            config._attn_implementation = 'sdpa'
            cpu = Qwen3ForCausalLM(config)
            reference = copy.deepcopy(cpu).cuda().train()
            if options.loss_chunk_size:
                from training.fsdp_loss import install_chunked_qwen_loss
                install_chunked_qwen_loss(cpu,chunk_size=options.loss_chunk_size,
                    backward_impl=options.loss_backward_impl)
            if options.ce_only_chunk_size:
                from training.ce_only_loss import install_ce_only_qwen_loss
                install_ce_only_qwen_loss(cpu,chunk_size=options.ce_only_chunk_size)
            from contextlib import ExitStack
            operators=ExitStack()
            if options.swiglu:
                from training.hf_operators import training_operators
                operators.enter_context(training_operators(cpu,swiglu=True,hf_semantics='triton_backward'))
            if options.prefetch:
                from training.prefetch_fsdp import shard_qwen_prefetched
                model=shard_qwen_prefetched(cpu).train()
            elif options.backend=='custom': model = shard_qwen(cpu).train()
            else:
                from training.official_fsdp import shard_official_qwen
                model = shard_official_qwen(cpu,keep_full_layers=options.keep_full_layers,
                    checkpoint_every=options.checkpoint_every,checkpoint_mode=options.checkpoint_mode).train()
            exchange_callbacks=[]
            if options.exchange_reduce:
                from torch.distributed.fsdp import FSDPModule
                from benchmarks.fsdp_bf16_exchange_adapter import BF16ExchangeReduceScatter
                for unit in model.modules():
                    if isinstance(unit,FSDPModule):
                        callback=BF16ExchangeReduceScatter(original_bf16=True,
                            backend=options.exchange_reduce,debug=True)
                        unit.set_custom_reduce_scatter(callback)
                        exchange_callbacks.append(callback)
            for module in model.modules():
                if isinstance(module, BlockFSDP):
                    module.compute_dtype = dtype
                    module.reduce_average = options.reduce_average
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
            opt_ref = torch.optim.AdamW(reference.parameters(), lr=1e-3, foreach=False)
            for step in range(3):
                torch.manual_seed(400 + dist.get_rank()*20 + step)
                ids = torch.randint(0, 97, (2, options.seq), device='cuda')
                opt.zero_grad(set_to_none=True); opt_ref.zero_grad(set_to_none=True)
                if options.accumulation==1:
                    actual=model(ids,labels=ids,use_cache=False)
                else:
                    from training.fsdp_accumulation import gradient_sync,sync_this_microbatch
                    outputs=[];total=torch.zeros((),device='cuda')
                    for i,micro in enumerate(ids.chunk(options.accumulation)):
                        with gradient_sync(model,sync_this_microbatch(i,options.accumulation,options.gradient_sync)):
                            part=model(micro,labels=micro,use_cache=False)
                            (part.loss/options.accumulation).backward()
                            outputs.append(part.logits.detach());total+=part.loss.detach()/options.accumulation
                    actual=SimpleNamespace(logits=torch.cat(outputs),loss=total)
                # Match per-unit BF16 casts: tied embed/head share an FP32
                # master, but each use casts separately before backward.
                shared_native = options.backend=='official' and tied
                from benchmarks.fsdp_reference import native_microbatch_reference
                expected = native_microbatch_reference(reference,ids,dtype=dtype,
                    tied=shared_native,accumulation=options.accumulation)
                if options.accumulation>1 and step==0:
                    # Diagnostic only: changing B2 -> B1 also changes native
                    # BF16 GEMM rounding, before any collective is involved.
                    with torch.no_grad():
                        full = native_microbatch_reference(reference,ids,dtype=dtype,tied=shared_native)
                    if dist.get_rank()==0:
                        print(json.dumps(dict(diagnostic='native_microbatch_vs_full',
                            max_logits_abs=(expected.logits.float()-full.logits.float()).abs().max().item(),
                            scope='Native-only batch-shape difference; not correctness tolerance')),flush=True)
                if options.loss_chunk_size:
                    assert actual.logits is None
                else:
                    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-4)
                torch.testing.assert_close(actual.loss, expected.loss)
                if options.accumulation==1:actual.loss.backward()
                expected.loss.backward()
                for p in reference.parameters():
                    dist.all_reduce(p.grad); p.grad.div_(dist.get_world_size())
                if options.backend=='official':
                    from training.fsdp_activation import canonical_parameter_name
                    for name,p in model.named_parameters():
                        target=reference.get_parameter(canonical_parameter_name(name)).grad
                        torch.testing.assert_close(p.grad.full_tensor(),target,
                            atol=3e-4 if tied else 2e-5,rtol=.03 if tied else 2e-3,
                            msg=lambda message:f'Parameter {name}, tied={tied}, step={step}: {message}')
                for name, unit in model.named_modules():
                    if not isinstance(unit, BlockFSDP): continue
                    ref_unit = reference.get_submodule(name)
                    target = torch.cat([p.grad.reshape(-1) for p in ref_unit.parameters()])
                    # Gradients/master values must be inspected without BF16 casting.
                    old_dtype = unit.compute_dtype; unit.compute_dtype = torch.float32
                    observed = unit.gather(unit.shard.grad)[:unit.numel]
                    unit.compute_dtype = old_dtype
                    torch.testing.assert_close(observed, target, atol=2e-5, rtol=2e-3)
                opt.step(); opt_ref.step()
                if options.backend=='official':
                    for name,p in model.named_parameters():
                        torch.testing.assert_close(p.full_tensor(),reference.get_parameter(canonical_parameter_name(name)),
                            atol=3e-4 if tied else 3e-5,rtol=.03 if tied else 3e-3)
                if options.checkpoint_test and step == 0:
                    from training.sharded_checkpoint import save_sharded, load_sharded
                    location = [tempfile.mkdtemp(prefix='fsdp-checkpoint-') if dist.get_rank()==0 else None]
                    dist.broadcast_object_list(location,src=0)
                    checkpoint = Path(location[0])/'step1'
                    metadata = dict(tied=tied,dtype=str(dtype))
                    save_sharded(checkpoint,model,opt,1,metadata)
                    with torch.no_grad():
                        for p in model.parameters(): p.add_(5)
                    opt.state.clear()
                    assert load_sharded(checkpoint,model,opt,metadata) == 1
                for name, unit in model.named_modules():
                    if not isinstance(unit, BlockFSDP): continue
                    target = torch.cat([p.detach().reshape(-1) for p in reference.get_submodule(name).parameters()])
                    old_dtype = unit.compute_dtype; unit.compute_dtype = torch.float32
                    observed = unit.gather(unit.shard.detach())[:unit.numel]
                    unit.compute_dtype = old_dtype
                    torch.testing.assert_close(observed, target, atol=3e-5, rtol=3e-3)
            if options.export_test:
                from training.export_sharded_hf import export_hf
                location = [tempfile.mkdtemp(prefix='fsdp-export-') if dist.get_rank()==0 else None]
                dist.broadcast_object_list(location,src=0)
                destination = Path(location[0])/'hf'
                export_hf(model,destination,Path(location[0]))
                reloaded = Qwen3ForCausalLM.from_pretrained(destination,torch_dtype=dtype,
                    attn_implementation='sdpa').cuda().eval()
                with torch.no_grad():
                    model.eval()
                    torch.testing.assert_close(model(ids,use_cache=False).logits,
                        reloaded(ids,use_cache=False).logits,atol=2e-5,rtol=2e-4)
                del reloaded
            if options.exchange_reduce:
                assert sum(c.calls for c in exchange_callbacks)>0, 'Candidate callback never executed'
            if dist.get_rank() == 0:
                print(json.dumps(dict(passed=True,tied=tied,backend=options.backend,dtype=str(dtype),world=dist.get_world_size(),
                    exchange_reduce=options.exchange_reduce,
                    exchange_callback_calls=sum(c.calls for c in exchange_callbacks),
                    allocator_backend=torch.cuda.get_allocator_backend(),
                    allocator_environment={k:os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')},
                    checkpoint=options.checkpoint_test,export=options.export_test,
                    prefetch=options.prefetch,keep_full_layers=options.keep_full_layers,
                    swiglu=options.swiglu,head_dim=options.head_dim,
                    accumulation=options.accumulation,gradient_sync=options.gradient_sync,
                    deterministic=options.deterministic,
                    ce_only_chunk_size=options.ce_only_chunk_size,
                    checkpoint_every=options.checkpoint_every,checkpoint_mode=options.checkpoint_mode,
                    loss_chunk_size=options.loss_chunk_size,loss_backward_impl=options.loss_backward_impl,
                    reference_microbatches=options.accumulation,
                    seq=options.seq,steps=3)),flush=True)
            operators.close()
            del model, reference, opt, opt_ref
    finally:
        dist.destroy_process_group()


if __name__ == '__main__': main()
