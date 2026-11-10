"""Fused Triton forward + full-gradient chunk-recomputed backward prototype."""
import torch
from ops.triton.linear_ce_forward import linear_ce_forward
from training.chunked_loss import _ChunkedLinearCE


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,hidden,weight,labels,chunk_size,ignore_index):
        count=(labels!=ignore_index).sum()
        ctx.save_for_backward(hidden,weight,labels,count)
        ctx.chunk_size,ctx.ignore_index=chunk_size,ignore_index
        return linear_ce_forward(hidden,weight,labels,ignore_index)

    @staticmethod
    def backward(ctx,grad_loss):
        return _ChunkedLinearCE.backward(ctx,grad_loss)


def fused_linear_cross_entropy(hidden,weight,labels,*,chunk_size=128,ignore_index=-100):
    return _FusedLinearCE.apply(hidden,weight,labels,chunk_size,ignore_index)
