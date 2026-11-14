"""Experimental exact-contribution vocabulary embedding; no production enablement."""
import torch
import torch.distributed as dist
from ops.triton.gradient_transport import reduce_bf16_fp32
class _VocabEmbedding(torch.autograd.Function):
 @staticmethod
 def forward(ctx,ids,weight,padding_idx,owner):
  owner.forward_calls+=1;ctx.owner=owner
  world=dist.get_world_size();rank=dist.get_rank();count=ids.numel();rows,width=weight.shape
  gathered=torch.empty(world*count,device=ids.device,dtype=ids.dtype)
  dist.all_gather_into_tensor(gathered,ids.contiguous().view(-1))
  owner=(gathered>=rank*rows)&(gathered<(rank+1)*rows)
  local_ids=(gathered-rank*rows).clamp(0,rows-1)
  partial=torch.nn.functional.embedding(local_ids,weight.to(torch.bfloat16))
  partial.masked_fill_(~owner[:,None],0)
  output=torch.empty((count,width),device=ids.device,dtype=torch.bfloat16)
  dist.reduce_scatter_tensor(output,partial,op=dist.ReduceOp.SUM)
  ctx.save_for_backward(gathered);ctx.shape=ids.shape;ctx.rows=rows;ctx.width=width;ctx.padding_idx=padding_idx
  return output.view(*ids.shape,width)
 @staticmethod
 def backward(ctx,dy):
  ctx.owner.backward_calls+=1
  ids,=ctx.saved_tensors;world=dist.get_world_size();rank=dist.get_rank();count=ids.numel()//world
  gathered=torch.empty((world*count,ctx.width),device=dy.device,dtype=torch.bfloat16)
  dist.all_gather_into_tensor(gathered,dy.contiguous().view(count,ctx.width))
  partials=torch.empty((world,ctx.rows,ctx.width),device=dy.device,dtype=torch.bfloat16)
  for r in range(world):
   full=torch.ops.aten.embedding_dense_backward(gathered[r*count:(r+1)*count],ids[r*count:(r+1)*count],world*ctx.rows,ctx.padding_idx,False)
   partials[r].copy_(full[rank*ctx.rows:(rank+1)*ctx.rows]);del full
  dw=torch.empty((ctx.rows,ctx.width),device=dy.device,dtype=torch.float32)
  reduce_bf16_fp32(partials.view(-1),dw,world,True)
  return None,dw,None,None
class VocabParallelEmbedding(torch.nn.Module):
 def __init__(self,full_weight_cpu,padding_idx=None):
  super().__init__();world=dist.get_world_size();rank=dist.get_rank()
  if full_weight_cpu.device.type!='cpu' or full_weight_cpu.shape[0]%world:raise ValueError('Equal CPU vocabulary shards required')
  self.forward_calls=0;self.backward_calls=0
  self.num_embeddings,self.embedding_dim=full_weight_cpu.shape;self.padding_idx=padding_idx
  rows=self.num_embeddings//world
  self.weight=torch.nn.Parameter(full_weight_cpu[rank*rows:(rank+1)*rows].float().cuda().clone())
 def forward(self,ids):return _VocabEmbedding.apply(ids,self.weight,-1 if self.padding_idx is None else self.padding_idx,self)
