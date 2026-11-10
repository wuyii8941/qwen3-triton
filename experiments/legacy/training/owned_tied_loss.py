"""Experimental first-order tied loss with an explicitly owned dW buffer.

The body and CE may compile separately; the enclosing reentrant autograd shell
is not a fullgraph model replacement. No gradient inputs are modified. dW is
created only after body backward, then updated in place while exclusively
owned here. Single-device dense training, no hooks/DDP/checkpointing/double-bwd.
"""
import torch
import triton
from training.tied_embedding import _merge


class _OwnedTiedLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, ids, labels, weight, *params):
        ctx.save_for_backward(weight,ids)
        embedding=torch.nn.functional.embedding(ids,weight).detach().requires_grad_(True)
        with torch.enable_grad():
            hidden=owner.body(embedding)
        logits=torch.nn.functional.linear(hidden.detach(),weight).detach().requires_grad_(True)
        with torch.enable_grad():
            loss=owner.ce(logits,labels)
        ctx.embedding,ctx.hidden,ctx.logits,ctx.loss=embedding,hidden,logits,loss
        ctx.params=params
        return loss.detach()

    @staticmethod
    def backward(ctx, grad_loss):
        if torch.is_grad_enabled():raise RuntimeError('Owned loss supports first-order backward only')
        weight,ids=ctx.saved_tensors
        with torch.enable_grad():
            grad_logits,=torch.autograd.grad(ctx.loss,ctx.logits,grad_loss,create_graph=False)
        # Release CE saved tensors before the transformer backward.
        ctx.loss=ctx.logits=None
        hidden=ctx.hidden.detach()
        grad_hidden=torch.matmul(grad_logits,weight)
        with torch.enable_grad():
            grads=torch.autograd.grad(ctx.hidden,(ctx.embedding,*ctx.params),grad_hidden,create_graph=False)
        ctx.hidden=ctx.embedding=ctx.params=None
        del grad_hidden
        grad_embedding,*grad_params=grads
        # Unlike an incoming autograd gradient, this fresh GEMM result is ours.
        dw=grad_logits.reshape(-1,weight.shape[0]).T @ hidden.reshape(-1,weight.shape[1])
        del grad_logits,hidden
        ordered,order=torch.sort(ids.reshape(-1),stable=True)
        dy=grad_embedding.contiguous().reshape(ids.numel(),weight.shape[1])
        _merge[(ids.numel(),)](dw,ordered,order,dy,dw,ids.numel(),weight.shape[1],
                              triton.next_power_of_2(weight.shape[1]))
        return None,None,None,dw,*grad_params


class OwnedTiedLoss:
    def __init__(self,model,*,compile_regions=True):
        self.model=model
        self.weight=model.model.embed_tokens.weight
        if self.weight is not model.lm_head.weight:raise ValueError('Tied weights required')
        embed=model.model.embed_tokens
        if (embed.padding_idx is not None or embed.max_norm is not None or embed.sparse
                or embed.scale_grad_by_freq or model.lm_head.bias is not None):
            raise ValueError('Only plain dense bias-free embedding/head supported')
        self.params=tuple(p for p in model.parameters() if p is not self.weight and p.requires_grad)
        def body(x):return model.model(inputs_embeds=x,use_cache=False).last_hidden_state
        def ce(logits,labels):return model.loss_function(logits=logits,labels=labels,vocab_size=model.config.vocab_size)
        self.body=torch.compile(body,fullgraph=True,dynamic=False) if compile_regions else body
        self.ce=torch.compile(ce,fullgraph=True,dynamic=False) if compile_regions else ce

    def __call__(self,ids,labels=None):
        if not self.model.training or not torch.is_grad_enabled():
            raise ValueError('Owned loss supports grad-enabled training only; evaluate with original model')
        if self.model.is_gradient_checkpointing:raise ValueError('Checkpointing not validated')
        if any(p._backward_hooks for p in self.model.parameters()):raise ValueError('Parameter gradient hooks unsupported')
        if not self.weight.requires_grad:raise ValueError('Trainable shared weight required')
        if ids.device.type!='cuda' or ids.ndim!=2 or not ids.numel():raise ValueError('Positive CUDA [B,S] input IDs required')
        return _OwnedTiedLoss.apply(self,ids,ids if labels is None else labels,self.weight,*self.params)
