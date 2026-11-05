"""Opt-in HF 4.51 training SDPA GQA: retain original Qwen projections/norm/RoPE.

Only remove repeat_kv/materialization and let native Flash SDPA map KV heads.
No global Transformers registry edits. Cache/sliding window are out of scope.
"""
from types import MethodType
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


def _forward(self, hidden_states, position_embeddings, attention_mask=None,
             past_key_value=None, cache_position=None, **kwargs):
    if past_key_value is not None or kwargs.get('output_attentions',False):
        raise ValueError('Native GQA training experiment does not support cache/attention outputs')
    shape=hidden_states.shape[:-1]
    heads=(*shape,-1,self.head_dim)
    q=self.q_norm(self.q_proj(hidden_states).view(heads)).transpose(1,2)
    k=self.k_norm(self.k_proj(hidden_states).view(heads)).transpose(1,2)
    v=self.v_proj(hidden_states).view(heads).transpose(1,2)
    cos,sin=position_embeddings
    q,k=apply_rotary_pos_emb(q,k,cos,sin)
    mask=attention_mask
    if mask is not None and mask.ndim==4: mask=mask[:,:,:,:k.shape[-2]]
    # Preserve the baseline's contiguous-layout boundary. The first strided
    # prototype changed compiled numerics too much; do not silently accept it.
    out=torch.nn.functional.scaled_dot_product_attention(q.contiguous(),k.contiguous(),v.contiguous(),attn_mask=mask,
        is_causal=q.shape[2]>1 and mask is None,enable_gqa=True,
        dropout_p=self.attention_dropout if self.training else 0.,scale=self.scaling)
    return self.o_proj(out.transpose(1,2).contiguous().reshape(*shape,-1)),None


def patch_native_gqa(model):
    saved=[]
    for layer in model.model.layers:
        module=layer.self_attn
        if module.sliding_window is not None:
            raise ValueError('Sliding-window attention unsupported')
        saved.append((module,module.__dict__.get('forward')))
        module.forward=MethodType(_forward,module)
    def undo():
        for module,old in reversed(saved):
            if old is None: del module.forward
            else: module.forward=old
    return undo
