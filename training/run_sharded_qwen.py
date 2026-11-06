"""Real pretrained Qwen3 multi-GPU full-parameter training smoke/deployment."""
import argparse
import os
import time
import json
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM
from training.block_fsdp import shard_qwen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--seq', type=int, default=128)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--data', type=Path, help='Existing uint32 token corpus; no downloading')
    p.add_argument('--checkpoint', type=Path, help='New checkpoint directory; requires ~12 bytes/global parameter free')
    p.add_argument('--resume', type=Path, help='Same-world-size sharded checkpoint')
    p.add_argument('--random-init', action='store_true', help='Capacity test only, NOT pretrained deployment')
    p.add_argument('--reference-check', action='store_true', help='Initial logits parity vs full unsharded HF BF16 model')
    p.add_argument('--export-hf', type=Path, help='New BF16 HF model directory (not optimizer checkpoint)')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if min(args.steps,args.seq,args.batch) < 1: p.error('Positive steps/seq/batch required')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(4)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    try:
        rank = dist.get_rank()
        torch.manual_seed(42)
        if args.random_init:
            config = AutoConfig.from_pretrained(args.model,local_files_only=True)
            model = AutoModelForCausalLM.from_config(config,torch_dtype=torch.bfloat16,
                attn_implementation='sdpa')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True)
        parameter_count = sum(p.numel() for p in model.parameters())
        if any(not p.requires_grad for p in model.parameters()):
            raise ValueError('This entry point requires full-parameter training')
        state_floor = parameter_count*16/dist.get_world_size()
        if state_floor > torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory-2*2**30:
            raise ValueError('FP32 master/gradient/AdamW states cannot fit; increase GPU count')
        model = shard_qwen(model).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
        protocol = dict(model=str(Path(args.model).resolve()),seq=args.seq,batch=args.batch,
            data=str(args.data.resolve()) if args.data else None,lr=1e-5,random_init=args.random_init)
        initial_step = 0
        if args.resume:
            from training.sharded_checkpoint import load_sharded
            initial_step = load_sharded(args.resume,model,opt,protocol)
        torch.manual_seed(1000 + rank)
        ids = torch.randint(0, model.config.vocab_size, (args.batch,args.seq), device='cuda')
        data = None
        if args.data:
            from training.flashoptim_convergence_job import Blocks
            data = Blocks(args.data,args.seq,training=True,seed=42)
            if len(data) < (initial_step+args.steps)*dist.get_world_size()*args.batch:
                raise ValueError('Corpus too small for nonoverlapping rank batches')
        parity = None
        if args.reference_check:
            if args.random_init or args.resume:
                raise ValueError('Reference check requires original pretrained weights, not random or resumed weights')
            if data is not None:
                ids = torch.stack([data[rank*args.batch+i] for i in range(args.batch)]).cuda()
            with torch.no_grad():
                model.eval()
                observed = model(ids,use_cache=False).logits
                reference = AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
                    torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).cuda().eval()
                expected = reference(ids,use_cache=False).logits
                parity = dict(max_abs=(observed.float()-expected.float()).abs().max().item())
                torch.testing.assert_close(observed,expected,atol=.03,rtol=.01)
                del reference,observed,expected
                model.train()
            torch.cuda.empty_cache()
            print(json.dumps(dict(rank=rank,initial_hf_parity=parity)),flush=True)
        records = []
        for step in range(initial_step,initial_step+args.steps):
            if data is not None:
                start_index = (step*dist.get_world_size()+rank)*args.batch
                ids = torch.stack([data[start_index+i] for i in range(args.batch)]).cuda()
            opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); start = time.perf_counter()
            loss = model(ids, labels=ids, use_cache=False).loss
            loss.backward()
            finite = torch.tensor(int(torch.isfinite(loss).item() and all(
                p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())),device='cuda')
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item(): raise RuntimeError('Nonfinite loss/gradient or missing gradient')
            opt.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            mean_loss = loss.detach().clone(); dist.all_reduce(mean_loss); mean_loss /= dist.get_world_size()
            record = dict(step=step+1,loss=mean_loss.item(),seconds=elapsed,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            records.append(record)
            print(json.dumps(dict(rank=rank,**record)),flush=True)
        args.output.mkdir(parents=True,exist_ok=True)
        if args.checkpoint:
            from training.sharded_checkpoint import save_sharded
            save_sharded(args.checkpoint,model,opt,initial_step+args.steps,protocol)
        if args.export_hf:
            from training.export_sharded_hf import export_hf
            # Gradients are no longer needed; preserve FP32 optimizer/master state.
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.eval()
                expected = model(ids,use_cache=False).logits[:,-1].detach().cpu()
                if rank == 0:
                    torch.save(dict(input_ids=ids.cpu(),last_logits=expected),
                               args.output/'export_reference.pt')
                del expected
            export_hf(model,args.export_hf,args.model)
        (args.output/f'rank{rank}.json').write_text(json.dumps(dict(completed=True,
            model=args.model,random_init=args.random_init,world_size=dist.get_world_size(),seq=args.seq,batch=args.batch,
            data=str(args.data) if args.data else 'rank-distinct fixed synthetic IDs; smoke only',
            initial_hf_parity=parity,export_hf=str(args.export_hf) if args.export_hf else None,
            global_parameters=parameter_count,full_parameter_training=True,
            local_parameters=sum(p.numel() for p in model.parameters()),records=records),indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__': main()
