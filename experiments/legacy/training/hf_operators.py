"""Scoped trainable HF ablations. Original weights and attention remain native.

Existing training kernels have different low-precision cast boundaries from
HF; numerical gates are mandatory, and these are NOT default replacements.
"""
from contextlib import contextmanager
from types import MethodType
from transformers.models.qwen3 import modeling_qwen3 as hf
from ops.triton.rms_norm import rms_norm_triton
from ops.triton.silu_and_mul import silu_and_mul_triton
from ops.triton.fused_qk_norm_rope import fused_qk_norm_rope_triton
import torch


@contextmanager
def training_operators(model, *, rms=False, swiglu=False, qk=False,hf_semantics=False):
    saved=[]
    def norm(self,x):
        if hf_semantics:
            from training.hf_rounded import norm as rounded_norm
            return rounded_norm(x,self.weight,self.variance_epsilon)
        return rms_norm_triton(x,self.weight,self.variance_epsilon)
    def mlp(self,x):
        if hf_semantics:
            from training.hf_rounded import activation,activation_triton_backward
            if hf_semantics=='triton_backward':activation=activation_triton_backward
            return self.down_proj(activation(self.gate_proj(x),self.up_proj(x)))
        return self.down_proj(silu_and_mul_triton(self.gate_proj(x),self.up_proj(x)))
    def attention(self,hidden_states,position_embeddings,attention_mask=None,
                  past_key_value=None,cache_position=None,**kwargs):
        if past_key_value is not None or kwargs.get('output_attentions',False):
            raise ValueError('Training QK ablation supports no cache/attention output')
        b,s,_=hidden_states.shape
        q=self.q_proj(hidden_states).view(b,s,-1,self.head_dim)
        k=self.k_proj(hidden_states).view(b,s,-1,self.head_dim)
        v=self.v_proj(hidden_states).view(b,s,-1,self.head_dim).transpose(1,2)
        cos,sin=position_embeddings
        q,k=fused_qk_norm_rope_triton(q,k,self.q_norm.weight,self.k_norm.weight,
            cos.expand(b,-1,-1).contiguous(),sin.expand(b,-1,-1).contiguous(),self.q_norm.variance_epsilon)
        out=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v,
            attn_mask=attention_mask,is_causal=s>1 and attention_mask is None,
            enable_gqa=True,dropout_p=0.,scale=self.scaling)
        return self.o_proj(out.transpose(1,2).reshape(b,s,-1)),None
    try:
        for module in model.modules():
            fn=norm if rms and isinstance(module,hf.Qwen3RMSNorm) else (
                mlp if swiglu and isinstance(module,hf.Qwen3MLP) else None)
            if qk and isinstance(module,hf.Qwen3Attention):fn=attention
            if fn:
                saved.append((module,module.__dict__.get('forward')))
                module.forward=MethodType(fn,module)
        yield
    finally:
        for module,old in reversed(saved):
            if old is None:del module.forward
            else:module.forward=old
