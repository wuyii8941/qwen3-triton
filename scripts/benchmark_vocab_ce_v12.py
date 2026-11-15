"""Four-rank official 8B head-only correctness and performance screen.

Not a full-model qualification. The reference uses native FSDP2 Linear/FP32
CE, with retained forward weights, FP32 gradients, matching the V11 strategy.
"""
from pathlib import Path
import argparse,json,os,statistics,time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard,MixedPrecisionPolicy
from safetensors import safe_open
from qwen3_triton.vocab_parallel import VocabParallelCE
from qwen3_triton.audit import NoFallback


def native_loss(model,h,ids):
    z=model(h)
    loss=torch.nn.functional.cross_entropy(z[:,:-1].float().reshape(-1,z.shape[-1]),ids[:,1:].reshape(-1),ignore_index=-100)
    return loss,z


def compare(a,b):
    a=a.detach().float();b=b.detach().float()
    return dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-8)),max_abs=float((a-b).abs().max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2);torch.manual_seed(20261115)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    dist.init_process_group('nccl',device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    try:
        assert dist.get_world_size()==4
        if rank==0:args.output.mkdir(exist_ok=False)
        dist.barrier()
        model_path=Path('/data1/tzh/models/Qwen/Qwen3-8B')
        index=json.loads((model_path/'model.safetensors.index.json').read_text())
        with safe_open(model_path/index['weight_map']['lm_head.weight'],framework='pt',device='cpu') as f:
            weight=f.get_tensor('lm_head.weight')
        assert weight.shape==(151936,4096) and weight.dtype==torch.bfloat16
        candidate=VocabParallelCE(weight)
        with torch.device('meta'):reference=torch.nn.Linear(4096,151936,bias=False)
        reference.weight=torch.nn.Parameter(weight.float());del weight
        fully_shard(reference,mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32),reshard_after_forward=False)
        torch.manual_seed(815+rank)
        h=torch.randn((1,512,4096),device='cuda',dtype=torch.bfloat16).requires_grad_()
        ids=torch.randint(0,151936,(1,512),device='cuda',dtype=torch.int64)
        ids[:,3+rank]=-100;ids[:,20+rank]=-100
        # Check every boundary, ignored target, and distinct rank inputs/counts.
        ids[:,1:5]=torch.tensor([0,37983,37984,151935],device='cuda')
        loss,z=native_loss(reference,h,ids);loss.backward()
        ref_loss=loss.detach().clone();ref_z=z.detach().clone();ref_dx=h.grad.detach().clone()
        ref_dw=reference.weight.grad.to_local().detach().clone();del z,loss
        reference.zero_grad(set_to_none=True);h.grad=None
        with NoFallback():loss,z=candidate(h,ids,diagnostics=True)
        with NoFallback(backward=True):loss.backward()
        rows=dict(rank=rank,loss_abs=float((loss-ref_loss).abs()),logits=compare(z,ref_z),
                  dx=compare(h.grad,ref_dx),dw=compare(candidate.weight.grad,ref_dw),
                  candidate_weight_grad_FP32=candidate.weight.grad.dtype==torch.float32)
        assert rows['loss_abs']<2e-5 and rows['logits']['relative_l2']<.01,rows
        assert rows['dx']['relative_l2']<.03 and rows['dw']['relative_l2']<.01,rows
        del z,loss,ref_z,ref_dx,ref_dw;candidate.zero_grad(set_to_none=True);h.grad=None
        # Non-unit, rank-dependent upstream scales exercise both collective adjoints.
        scale=1.+rank/4
        loss,_=native_loss(reference,h,ids);(loss*scale).backward()
        ref_dx=h.grad.detach().clone();ref_dw=reference.weight.grad.to_local().detach().clone()
        reference.zero_grad(set_to_none=True);h.grad=None
        loss,_=candidate(h,ids);(loss*scale).backward()
        rows['scaled_dx']=compare(h.grad,ref_dx);rows['scaled_dw']=compare(candidate.weight.grad,ref_dw)
        assert rows['scaled_dx']['relative_l2']<.03 and rows['scaled_dw']['relative_l2']<.01,rows
        del loss,ref_dx,ref_dw;reference.zero_grad(set_to_none=True);candidate.zero_grad(set_to_none=True);h.grad=None
        timings={}
        for name in ['HF_r1','Triton_r1','Triton_r2','HF_r2']:
            measured=[];module=reference if name.startswith('HF') else candidate
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            for step in range(11):
                module.zero_grad(set_to_none=True);h.grad=None
                torch.cuda.synchronize();dist.barrier();start=time.perf_counter()
                if module is reference:loss,_=native_loss(module,h,ids)
                else:loss,_=module(h,ids)
                loss.backward();torch.cuda.synchronize();elapsed=time.perf_counter()-start
                elapsed_tensor=torch.tensor(elapsed,device='cuda',dtype=torch.float64)
                dist.all_reduce(elapsed_tensor,op=dist.ReduceOp.MAX)
                if step>=3:measured.append(float(elapsed_tensor))
                del loss
            timings[name]=dict(median=statistics.median(measured),samples=measured,
                               peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                               peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
            if rank==0:print(json.dumps(dict(event=name,**timings[name])),flush=True)
        rows['timing']=timings
        (args.output/f'rank{rank}.json').write_text(json.dumps(rows,indent=2)+'\n')
        gathered=[None]*4;dist.all_gather_object(gathered,rows)
        if rank==0:
            result=dict(complete=True,scope=__doc__,full_model_qualified=False,model='official Qwen3-8B LM head',
                        weight_shape=[151936,4096],input_shape_per_rank=[1,512,4096],world_size=4,
                        precision='BF16 compute; FP32 master, gradient accumulation and dX collective',ranks=gathered,
                        reductions=[1-timings[f'Triton_r{i}']['median']/timings[f'HF_r{i}']['median'] for i in (1,2)])
            (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(complete=True,reductions=result['reductions'],full_model_qualified=False)),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
