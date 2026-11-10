"""Batch-equivalent indexing, token-weighted loss, and explicit sync contexts.

The caller must place BOTH forward and backward inside gradient_sync. A
deferred FSDP reduction can retain full gradients; it is not memory-free.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F


def step_sample_indices(step,rank,world,microbatch,accumulation):
    if min(world,microbatch,accumulation)<1 or step<0 or not 0<=rank<world:
        raise ValueError('Invalid step/rank/batch dimensions')
    start=(step*world+rank)*microbatch*accumulation
    return [list(range(start+i*microbatch,start+(i+1)*microbatch)) for i in range(accumulation)]


def sync_this_microbatch(index,count,mode):
    if count<1 or not 0<=index<count or mode not in ('each','defer'):
        raise ValueError('Invalid accumulation sync configuration')
    return mode=='each' or index==count-1


@contextmanager
def gradient_sync(model,enabled):
    if enabled:
        yield
    elif hasattr(model,'set_requires_gradient_sync'):
        model.set_requires_gradient_sync(False)
        try:yield
        finally:model.set_requires_gradient_sync(True)
    elif hasattr(model,'no_sync'):
        with model.no_sync():yield
    else:
        raise ValueError('Deferred synchronization requires FSDP2 or DDP')


def shifted_valid_tokens(labels,ignore_index=-100):
    if labels.ndim!=2 or labels.shape[1]<2:raise ValueError('Expected labels [B,S] with S>=2')
    return (labels[:,1:]!=ignore_index).sum()


def global_weighted_causal_loss(logits,labels,*,global_tokens,world,ignore_index=-100):
    """Contribution to a global token mean under world-averaged gradients.

    global_tokens is the valid shifted-token total over ALL microbatches and
    ranks, computed once before the accumulation group. Unequal padding and
    an all-ignored local microbatch are supported; a globally empty group is
    rejected. The distributed mean of rank contributions is the scalar loss.
    """
    if global_tokens<1 or world<1:raise ValueError('Positive global token count/world required')
    if logits.ndim!=3 or labels.shape!=logits.shape[:2] or labels.shape[1]<2:
        raise ValueError('Expected logits [B,S,V] and labels [B,S]')
    total=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),
        labels[:,1:].reshape(-1),ignore_index=ignore_index,reduction='sum')
    return total*(world/global_tokens)
