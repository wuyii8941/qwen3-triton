"""EXPERIMENTAL chain-only asynchronous parameter prefetch.

Not enabled by the baseline or convergence runner. GPU correctness and latency
gates are required before adoption. NCCL collective order must match on ranks.
"""
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten
from training.block_fsdp import BlockFSDP
from training.minimal_fsdp import copy_scale


class ParameterPrefetch:
    def __init__(self,units):
        self.stream=torch.cuda.Stream()
        self.pending={}
        self.next={}
        for phase,order in [('forward',units),('backward',list(reversed(units)))]:
            self.next[phase]={unit:order[i+1] if i+1<len(order) else None
                              for i,unit in enumerate(order)}

    def request(self,unit,shard,phase):
        key=(unit,phase)
        if key in self.pending:return
        # In particular, wait for the previous optimizer step's writes before
        # reading master shards on the prefetch stream.
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            local=copy_scale(shard,dtype=unit.compute_dtype)
            full=torch.empty(unit.size*unit.world,device=shard.device,dtype=unit.compute_dtype)
            work=dist.all_gather_into_tensor(full,local,async_op=True)
        self.pending[key]=(full,local,work)

    def acquire(self,unit,shard,phase):
        self.request(unit,shard,phase)
        full,local,work=self.pending.pop((unit,phase))
        # Work.wait establishes the NCCL -> consumer-stream dependency; keep
        # both allocations alive until the consumer stream reaches that wait.
        work.wait()
        full.record_stream(torch.cuda.current_stream())
        local.record_stream(torch.cuda.current_stream())
        following=self.next[phase][unit]
        if following is not None:self.request(following,following.shard,phase)
        return full


class _PrefetchedCall(torch.autograd.Function):
    @staticmethod
    def forward(ctx,owner,structure,constants,shard,*inputs):
        ctx.owner,ctx.structure,ctx.constants=owner,structure,constants
        ctx.save_for_backward(shard,*inputs)
        args,kwargs=owner.unpack(structure,constants,inputs)
        full=owner.prefetch.acquire(owner,shard,'forward')
        output=owner.call(full,args,kwargs)
        ctx.tuple_output=isinstance(output,tuple)
        if ctx.tuple_output:
            if len(output)!=1:raise ValueError('Single tensor output required')
            output=output[0]
        return output

    @staticmethod
    def backward(ctx,grad):
        if torch.is_grad_enabled():raise RuntimeError('First-order backward only')
        owner=ctx.owner
        shard,*saved=ctx.saved_tensors
        with torch.enable_grad():
            full=owner.prefetch.acquire(owner,shard,'backward').requires_grad_(True)
            inputs=[x.detach().requires_grad_(x.requires_grad) for x in saved]
            differentiable=[x for x in inputs if x.requires_grad]
            args,kwargs=owner.unpack(ctx.structure,ctx.constants,inputs)
            output=owner.call(full,args,kwargs)
            if ctx.tuple_output:output=output[0]
            gradients=torch.autograd.grad(output,(full,*differentiable),grad)
        full_gradient=copy_scale(gradients[0].contiguous(),dtype=torch.float32)
        local=torch.empty_like(shard)
        # Match baseline SUM + FP32 scale; do not mix AVG or lower-precision
        # communication into this prefetch ablation.
        dist.reduce_scatter_tensor(local,full_gradient,op=dist.ReduceOp.SUM)
        iterator=iter(gradients[1:])
        input_gradients=[next(iterator) if x.requires_grad else None for x in inputs]
        return (None,None,None,copy_scale(local,1./owner.world),*input_gradients)


class PrefetchedBlockFSDP(BlockFSDP):
    def forward(self,*args,**kwargs):
        if kwargs.get('use_cache') or kwargs.get('output_attentions'):
            raise ValueError('Training without cache/attention outputs required')
        leaves,structure=tree_flatten((args,kwargs))
        constants=[(isinstance(x,torch.Tensor),None if isinstance(x,torch.Tensor) else x) for x in leaves]
        inputs=[x for x in leaves if isinstance(x,torch.Tensor)]
        output=_PrefetchedCall.apply(self,structure,constants,self.shard,*inputs)
        return (output,) if self._template.__class__.__name__.endswith('DecoderLayer') else output


def shard_qwen_prefetched(model):
    if getattr(model.config,'attention_dropout',0):raise ValueError('Dropout unsupported')
    shared={};units=[]
    def wrap(module):
        unit=PrefetchedBlockFSDP(module,shared=shared);units.append(unit);return unit
    body=model.model
    body.embed_tokens=wrap(body.embed_tokens)
    for i in range(len(body.layers)):body.layers[i]=wrap(body.layers[i])
    body.norm=wrap(body.norm)
    model.lm_head=wrap(model.lm_head)
    body.rotary_emb.cuda();model.config.use_cache=False
    engine=ParameterPrefetch(units)
    for unit in units:unit.prefetch=engine
    return model
