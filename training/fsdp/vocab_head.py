"""Experimental untied vocabulary-sharded LM head; not production enabled.

Each rank owns contiguous FP32 master rows. Global hidden and BF16 logits
are exchanged; full BF16 weights are gathered once for native exact dX.
Gradients already include the global data-parallel mean: exclude this
parameter from FSDP and do not reduce it again.
"""
import torch
import torch.distributed as dist
from ops.triton.gradient_transport import reduce_bf16_fp32

class _VocabLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,owner):
        owner.forward_calls+=1;ctx.owner=owner
        world=dist.get_world_size();rows=x.shape[0];hidden=x.shape[1];vocab=weight.shape[0]
        gathered=torch.empty((world*rows,hidden),device=x.device,dtype=x.dtype)
        dist.all_gather_into_tensor(gathered,x)
        gathered=gathered.view(world,rows,hidden);w=weight.to(torch.bfloat16)
        send=torch.empty((world,rows,vocab),device=x.device,dtype=x.dtype)
        for r in range(world):torch.mm(gathered[r],w.t(),out=send[r])
        received=torch.empty_like(send);dist.all_to_all_single(received,send)
        ctx.save_for_backward(gathered,w);ctx.world=world;ctx.rows=rows;ctx.vocab=vocab
        return received.transpose(0,1).contiguous().view(rows,world*vocab)
    @staticmethod
    def backward(ctx,dy):
        ctx.owner.backward_calls+=1
        gathered,w=ctx.saved_tensors;world=ctx.world;rows=ctx.rows;vocab=ctx.vocab
        outgoing=dy.view(rows,world,vocab).transpose(0,1).contiguous();incoming=torch.empty_like(outgoing)
        dist.all_to_all_single(incoming,outgoing)
        full_weight=torch.empty((world*vocab,w.shape[1]),device=dy.device,dtype=w.dtype)
        dist.all_gather_into_tensor(full_weight,w)
        dx=torch.mm(dy.contiguous(),full_weight)
        del full_weight
        partial_dw=torch.empty((world,vocab,w.shape[1]),device=dy.device,dtype=torch.bfloat16)
        for r in range(world):torch.mm(incoming[r].t(),gathered[r],out=partial_dw[r])
        dw=torch.empty(w.shape,device=dy.device,dtype=torch.float32)
        reduce_bf16_fp32(partial_dw.view(-1),dw,world,True)
        return dx.to(torch.bfloat16),dw,None

class VocabParallelHead(torch.nn.Module):
    def __init__(self, full_weight_cpu):
        super().__init__()
        if full_weight_cpu.device.type!='cpu' or full_weight_cpu.ndim!=2:
            raise ValueError('CPU full weight matrix required')
        world=dist.get_world_size();rank=dist.get_rank()
        if full_weight_cpu.shape[0]%world:raise ValueError('Equal vocabulary shards required')
        self.forward_calls=0;self.backward_calls=0
        self.out_features,self.in_features=full_weight_cpu.shape
        rows=self.out_features//world
        self.weight=torch.nn.Parameter(full_weight_cpu[rank*rows:(rank+1)*rows].float().cuda().clone())
    def forward(self,hidden):
        if hidden.dtype!=torch.bfloat16 or hidden.shape[-1]!=self.in_features:
            raise ValueError('BF16 hidden states with matching width required')
        shape=hidden.shape[:-1]
        return _VocabLinear.apply(hidden.reshape(-1,self.in_features),self.weight,self).reshape(*shape,self.out_features)
