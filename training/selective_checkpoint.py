"""Selective recomputation prototype: keep GEMM/Flash results, redo cheap ops."""
from functools import partial
import torch
from torch.utils.checkpoint import CheckpointPolicy,create_selective_checkpoint_contexts


def policy(ctx,op,*args,**kwargs):
    expensive=(torch.ops.aten.mm.default,torch.ops.aten.addmm.default,torch.ops.aten.bmm.default,
               torch.ops.aten._scaled_dot_product_flash_attention.default,
               torch.ops.aten._scaled_dot_product_efficient_attention.default)
    return CheckpointPolicy.MUST_SAVE if op in expensive else CheckpointPolicy.PREFER_RECOMPUTE


def enable_selective_checkpoint(model):
    if model.is_gradient_checkpointing:raise ValueError('Do not stack checkpoint policies')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False,
        context_fn=partial(create_selective_checkpoint_contexts,policy)))
    return model
