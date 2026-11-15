"""Experimental Triton vocabulary-parallel projection and exact full CE.

Ranks keep their data-parallel decoder inputs and contiguous FP32 vocabulary
master shards. Hidden states/labels are gathered, vocabulary-local CE uses
FP32 sufficient-statistic reductions, and dX uses FP32 ReduceScatter. Each
data-rank dW GEMM rounds to BF16, then contributions are averaged in FP32,
matching the existing BF16 compute / FP32 data-parallel-gradient policy.

The decoder remains FSDP with averaging; this head's gradient is already
averaged and MUST be excluded from FSDP. All ranks must call together, with
the same local shape. Dense logits are an explicit diagnostic option only.
NCCL collectives, allocation, casts and layout copies use PyTorch.
"""
import torch
import torch.distributed as dist
from .kernels.linear import matmul
from .kernels.attention import cast_tensor
from .kernels.vocab_ce import (local_statistics,packed_statistics,finish_statistics,
                               logit_gradient,accumulate_weight_gradient)


class _VocabCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,hidden,weight,labels,ignore,diagnostics,owner):
        world=dist.get_world_size();rank=dist.get_rank();b,s,k=hidden.shape;r=b*s;v=weight.shape[0]
        if (hidden.dtype!=torch.bfloat16 or weight.dtype!=torch.float32 or labels.dtype!=torch.int64
                or labels.shape!=(b,s) or s<2 or not hidden.is_contiguous() or not labels.is_contiguous()
                or hidden.device!=weight.device or hidden.device!=labels.device):
            raise ValueError('Matching contiguous BF16 hidden, int64 labels and local FP32 weight required')
        all_hidden=torch.empty((world*r,k),device=hidden.device,dtype=hidden.dtype)
        all_labels=torch.empty(world*r,device=hidden.device,dtype=labels.dtype)
        dist.all_gather_into_tensor(all_hidden,hidden.view(r,k))
        dist.all_gather_into_tensor(all_labels,labels.view(-1))
        w=cast_tensor(weight,torch.bfloat16)
        logits=matmul(all_hidden,w.t(),optimized=True)
        local_lse,target=local_statistics(logits,all_labels,s,rank*v)
        maximum=local_lse.clone();dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
        packed=packed_statistics(local_lse,maximum,target);dist.all_reduce(packed,op=dist.ReduceOp.SUM)
        lse,counts,means=finish_statistics(maximum,packed,all_labels,r,s,world,ignore)
        loss=means[rank].clone();dense=None
        if diagnostics:
            received=torch.empty_like(logits)
            dist.all_to_all_single(received,logits)
            dense=received.view(world,r,v).transpose(0,1).contiguous().view(b,s,world*v)
            ctx.mark_non_differentiable(dense)
        ctx.save_for_backward(all_hidden,w,all_labels,logits,lse,counts)
        ctx.rows=r;ctx.sequence=s;ctx.world=world;ctx.start=rank*v;ctx.ignore=ignore;ctx.owner=owner
        owner.forward_calls+=1
        return loss,dense

    @staticmethod
    def backward(ctx,up,_diagnostic_grad):
        h,w,labels,logits,lse,counts=ctx.saved_tensors
        world=ctx.world;r=ctx.rows;k=h.shape[1]
        ups=torch.empty(world,device=h.device,dtype=up.dtype)
        dist.all_gather_into_tensor(ups,up.contiguous().view(1))
        dz=logit_gradient(logits,labels,lse,counts,ups,r,ctx.sequence,ctx.start,ctx.ignore)
        partial_dx=matmul(dz,w,optimized=True,output_dtype=torch.float32)
        local_dx=torch.empty((r,k),device=h.device,dtype=torch.float32)
        dist.reduce_scatter_tensor(local_dx,partial_dx,op=dist.ReduceOp.SUM)
        dx=cast_tensor(local_dx,torch.bfloat16)
        if ctx.owner.fused_dw:
            from .kernels.vocab_dw import weight_gradient_mean
            dw=weight_gradient_mean(dz,h,r,world,ctx.owner.dw_config)
        else:
            dw=torch.empty(w.shape,device=w.device,dtype=torch.float32)
            for source in range(world):
                partial_dw=matmul(dz[source*r:(source+1)*r].t(),h[source*r:(source+1)*r],optimized=True)
                accumulate_weight_gradient(partial_dw,dw,world,source==0)
        ctx.owner.backward_calls+=1
        return dx.view(-1,ctx.sequence,k),dw,None,None,None,None


class VocabParallelCE(torch.nn.Module):
    def __init__(self,full_weight_cpu,*,fused_dw=False):
        super().__init__();world=dist.get_world_size();rank=dist.get_rank()
        if full_weight_cpu.device.type!='cpu' or full_weight_cpu.ndim!=2 or full_weight_cpu.shape[0]%world:
            raise ValueError('Equal CPU vocabulary shards required')
        self.out_features,self.in_features=full_weight_cpu.shape;v=self.out_features//world
        self.weight=torch.nn.Parameter(full_weight_cpu[rank*v:(rank+1)*v].float().cuda().clone())
        self.forward_calls=0;self.backward_calls=0
        self.fused_dw=bool(fused_dw)
        self.dw_config=(64,128,32,8,8,3) if self.fused_dw else None

    def forward(self,hidden,labels,*,ignore_index=-100,diagnostics=False):
        if hidden.ndim!=3 or hidden.shape[-1]!=self.in_features:
            raise ValueError('Expected [B,S,H] hidden states')
        return _VocabCE.apply(hidden,self.weight,labels,ignore_index,diagnostics,self)
