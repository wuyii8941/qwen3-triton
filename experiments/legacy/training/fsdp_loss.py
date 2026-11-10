"""Opt-in Qwen3 loss-only forward, installed BEFORE FSDP/optimizer creation.

The head must be called as a Module, not read through ``lm_head.weight``
outside its forward: FSDP's pre-forward hook owns parameter all-gather.
GPU/FSDP parity and performance remain separate adoption gates.
"""
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from training.chunked_loss import chunked_linear_cross_entropy


class LossLinear(nn.Linear):
    def forward(self, hidden, labels=None):
        if labels is None:
            return F.linear(hidden, self.weight, self.bias)
        if self.bias is not None:
            raise ValueError('LossLinear requires a bias-free head')
        return chunked_linear_cross_entropy(
            hidden.reshape(-1, hidden.shape[-1]), self.weight,
            labels.reshape(-1), chunk_size=self.loss_chunk_size,
            backward_impl=self.loss_backward_impl)


def install_chunked_qwen_loss(model, *, chunk_size=128, backward_impl='default'):
    """Preserve weight identity, tying and state keys; omit logits with labels.

    No-label inference retains the original HF forward. Training supports
    ordinary mean shifted CE only: no cache, tuple output, custom token-count
    denominator, or logits slicing. Unsupported semantics fail explicitly.
    Caller must install before sharding, hooks, compilation and optimizer.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if backward_impl not in ('default', 'native_ce', 'native_ce_padded'):
        raise ValueError('Unknown chunked CE backward implementation')
    from transformers import Qwen3ForCausalLM
    if not isinstance(model, Qwen3ForCausalLM):
        raise TypeError('Only HF Qwen3ForCausalLM is supported')
    if hasattr(model, '_chunked_original_forward'):
        raise ValueError('Chunked loss already installed')
    old = model.lm_head
    if type(old) is not nn.Linear or old.bias is not None:
        raise ValueError('Install on a plain bias-free Linear head before FSDP')
    if old._forward_hooks or old._forward_pre_hooks:
        raise ValueError('Install before registering head hooks')
    if any(type(p).__name__ == 'DTensor' for p in model.parameters()):
        raise ValueError('Install before FSDP')
    # Meta allocation does not duplicate a real vocabulary-sized weight.
    head = LossLinear(old.in_features, old.out_features, bias=False, device='meta')
    head.weight = old.weight
    head.loss_chunk_size = chunk_size
    head.loss_backward_impl = backward_impl
    head.train(old.training)
    model.lm_head = head
    model._chunked_original_forward = model.forward

    def forward(self, input_ids=None, *, labels=None, **kwargs):
        if labels is None:
            return self._chunked_original_forward(input_ids=input_ids, **kwargs)
        allowed = {'attention_mask', 'position_ids', 'inputs_embeds',
                   'output_attentions', 'output_hidden_states', 'cache_position',
                   'use_cache', 'past_key_values', 'return_dict', 'logits_to_keep'}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f'Unsupported loss-only arguments: {sorted(unknown)}')
        kwargs = dict(kwargs)
        if kwargs.pop('use_cache', False) not in (False, None):
            raise ValueError('Loss-only forward requires use_cache=False')
        if kwargs.pop('past_key_values', None) is not None:
            raise ValueError('Loss-only forward does not accept a cache')
        if kwargs.pop('return_dict', True) is not True:
            raise ValueError('Loss-only forward requires return_dict=True')
        keep = kwargs.pop('logits_to_keep', 0)
        if not isinstance(keep, int) or keep != 0:
            raise ValueError('Loss-only forward does not support logits slicing')
        # Always traverse model and head __call__, preserving FSDP hooks.
        outputs = self.model(input_ids=input_ids, use_cache=False,
                             return_dict=True, **kwargs)
        hidden = outputs.last_hidden_state
        if labels.shape != hidden.shape[:2] or hidden.shape[1] < 2:
            raise ValueError('Expected labels [B,S] with S >= 2')
        loss = self.lm_head(hidden[:, :-1].contiguous(),
                            labels=labels[:, 1:].contiguous())
        return CausalLMOutputWithPast(loss=loss, logits=None,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    model.forward = MethodType(forward, model)
    return model
