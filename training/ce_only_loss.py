"""Opt-in CE-only row chunking; native full LM-head GEMM remains untouched.

Unlike chunked_linear_cross_entropy, no linear GEMM is split/recomputed.
Full low-precision logits/gradient remain allocated. Only FP32 CE temporaries
are chunk bounded. First-order mean causal CE only; not a fused Triton loss.
"""
from functools import partial
import torch
from torch.nn import functional as F


class _ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logits,labels,chunk_size,ignore_index):
        count=(labels!=ignore_index).sum()
        total=torch.zeros((),device=logits.device,dtype=torch.float32)
        for start in range(0,len(labels),chunk_size):
            total+=F.cross_entropy(logits[start:start+chunk_size].float(),
                labels[start:start+chunk_size],ignore_index=ignore_index,reduction='sum')
        ctx.save_for_backward(logits,labels,count)
        ctx.chunk_size,ctx.ignore_index=chunk_size,ignore_index
        return total/count

    @staticmethod
    def backward(ctx,grad_loss):
        logits,labels,count=ctx.saved_tensors
        grad=torch.empty_like(logits)
        for start in range(0,len(labels),ctx.chunk_size):
            with torch.enable_grad():
                z=logits[start:start+ctx.chunk_size].detach().float().requires_grad_()
                loss=F.cross_entropy(z,labels[start:start+ctx.chunk_size],
                    ignore_index=ctx.ignore_index,reduction='sum')
                dz,=torch.autograd.grad(loss,z,grad_outputs=grad_loss.float()/count.clamp_min(1))
            grad[start:start+len(z)]=dz.to(logits.dtype)
        return grad,None,None,None


def chunked_causal_ce(logits,labels,vocab_size,*,chunk_size=128,ignore_index=-100,
                      num_items_in_batch=None,shift_labels=None,**kwargs):
    if kwargs or num_items_in_batch is not None:
        raise ValueError('CE-only prototype supports mean causal CE only')
    if type(chunk_size) is not int or chunk_size<1:
        raise ValueError('Positive integer chunk size required')
    if shift_labels is None:
        shift_labels=F.pad(labels,(0,1),value=ignore_index)[...,1:].contiguous()
    if logits.shape[-1]!=vocab_size or logits.shape[:-1]!=shift_labels.shape:
        raise ValueError('Full-shape logits and shifted labels required')
    return _ChunkedCE.apply(logits.reshape(-1,vocab_size),
        shift_labels.reshape(-1).to(logits.device),chunk_size,ignore_index)


def install_ce_only_qwen_loss(model,*,chunk_size=128):
    """Replace only HF loss callback, before FSDP; no head/parameter surgery."""
    from transformers import Qwen3ForCausalLM
    if not isinstance(model,Qwen3ForCausalLM):raise TypeError('Qwen3ForCausalLM required')
    if type(chunk_size) is not int or chunk_size<1:raise ValueError('Positive integer chunk size required')
    if hasattr(model,'_ce_only_original_loss_function') or hasattr(model,'_chunked_original_forward'):
        raise ValueError('Do not stack or duplicate loss adapters')
    if any(hasattr(p,'to_local') for p in model.parameters()):raise ValueError('Install before FSDP')
    model._ce_only_original_loss_function=model.loss_function
    model.loss_function=partial(chunked_causal_ce,chunk_size=chunk_size)
