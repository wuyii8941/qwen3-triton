"""Exact full-gradient, token-chunked linear CE with backward recomputation.

This first reference bounds logits storage by chunk_size * vocabulary. It
uses native GEMMs, no gradient filtering, and is not yet a fused Triton CCE.
Caller supplies shifted hidden states and labels; tied weights work normally.
"""
import torch
from torch.nn import functional as F


class _ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk_size, ignore_index):
        if hidden.ndim != 2 or weight.ndim != 2 or labels.shape != hidden.shape[:1]:
            raise ValueError('Expected hidden [N,H], weight [V,H], labels [N]')
        if hidden.shape[1] != weight.shape[1] or chunk_size < 1:
            raise ValueError('Invalid hidden dimension or chunk size')
        if hidden.dtype != weight.dtype:
            raise ValueError('Hidden and weight dtype must match')
        count = (labels != ignore_index).sum()
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for start in range(0, len(labels), chunk_size):
            logits = F.linear(hidden[start:start+chunk_size], weight).float()
            total += F.cross_entropy(logits, labels[start:start+chunk_size],
                                     ignore_index=ignore_index, reduction='sum')
        ctx.save_for_backward(hidden, weight, labels, count)
        ctx.chunk_size, ctx.ignore_index = chunk_size, ignore_index
        return total / count

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, weight, labels, count = ctx.saved_tensors
        # Native mean CE returns NaN for all-ignored loss but ZERO gradients.
        grad_count=count.clamp_min(1)
        dx = torch.empty_like(hidden) if ctx.needs_input_grad[0] else None
        # FP32 across chunks; cast only once after the complete reduction.
        dw = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        for start in range(0, len(labels), ctx.chunk_size):
            x = hidden[start:start+ctx.chunk_size]
            target = labels[start:start+ctx.chunk_size]
            logits = F.linear(x, weight).float()
            if hidden.is_cuda:
                from ops.triton.ce_gradient import ce_gradient
                grad_logits=ce_gradient(logits,logits.logsumexp(-1),target,grad_count,grad_loss,
                    dtype=hidden.dtype,ignore_index=ctx.ignore_index)
            else:
                prob = logits.softmax(-1)
                valid = target != ctx.ignore_index
                row = torch.arange(len(target), device=target.device)
                prob[row, target.masked_fill(~valid, 0)] -= valid.to(prob.dtype)
                prob *= valid[:, None] * (grad_loss.float() / grad_count)
                grad_logits = prob.to(hidden.dtype)
            if dx is not None:
                dx[start:start+len(target)] = grad_logits @ weight
            if dw is not None:
                # Accumulate into the persistent FP32 buffer directly instead
                # of allocating/writing/reading a full [V,H] temporary per chunk.
                dw.addmm_(grad_logits.float().T, x.float())
        return dx, None if dw is None else dw.to(weight.dtype), None, None, None


def chunked_linear_cross_entropy(hidden, weight, labels, *, chunk_size=128, ignore_index=-100):
    return _ChunkedLinearCE.apply(hidden, weight, labels, chunk_size, ignore_index)
