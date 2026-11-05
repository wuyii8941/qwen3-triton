"""PyTorch-only Qwen operators on the lab's static compact Flash cache.

This is NOT untouched HF generate(): attention/cache plumbing is adapted so
Torch and Triton paths share the same effective-length Flash contract. Norm,
RoPE, MLP and projections remain HF/PyTorch. No Triton replacement is enabled.
"""
from contextlib import contextmanager
from types import MethodType
import torch
import transformers
from transformers.models.qwen3 import modeling_qwen3 as hf
from model.hf_attention import make_attention_forward
from model.static_decode import static_mask_or_original

# Keep the reference pure Torch even when another model has an active scoped
# Triton adapter (that legacy adapter patches HF's module-global RoPE).
_original_rope = hf.apply_rotary_pos_emb


@contextmanager
def qwen3_torch_static_reference(model):
    if transformers.__version__!='4.51.0' or model.config.model_type!='qwen3':
        raise ValueError('Pinned Qwen3 / Transformers 4.51.0 required')
    if torch.is_grad_enabled() or model.training:
        raise ValueError('Use eval + inference_mode')
    if model.config._attn_implementation!='sdpa' or getattr(model.config,'use_sliding_window',False):
        raise ValueError('SDPA without sliding window required')
    saved=[]
    old_mask=model.model.__dict__.get('_update_causal_mask')
    fn=make_attention_forward(fuse_qk=False,compact_gqa=True,rope_fn=_original_rope)
    try:
        model.model._update_causal_mask=MethodType(
            static_mask_or_original(model.model._update_causal_mask),model.model)
        for module in model.modules():
            if isinstance(module,hf.Qwen3Attention):
                saved.append((module,module.__dict__.get('forward')))
                module.forward=MethodType(fn,module)
        yield
    finally:
        if old_mask is None: del model.model._update_causal_mask
        else: model.model._update_causal_mask=old_mask
        for module,old in reversed(saved):
            if old is None: del module.forward
            else: module.forward=old
