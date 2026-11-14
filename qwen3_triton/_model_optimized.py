"""Explicit optimization/ablation variants over the frozen complete baseline.

These variants are implementation candidates, not qualified performance claims.
All five switches map to actual distinct compute paths. No HF compute fallback.
"""
from dataclasses import dataclass,replace
from ._model import (TritonQwen3ForCausalLM,ComputeOptions,TritonAttention,TritonMLP)
from .kernels.norm_rope import norm_rope
from .kernels.mlp import gate_up_swiglu
from .kernels.attention import causal_attention

@dataclass(frozen=True)
class OptimizationOptions(ComputeOptions):
    linear_optimized: bool=True
    attention_optimized: bool=True
    loss_optimized: bool=True
    norm_rope_fusion: bool=True
    mlp_fusion: bool=True

class OptimizedAttention(TritonAttention):
    def forward(self,x,positions):
        if not self.options.norm_rope_fusion:return super().forward(x,positions)
        b,s,_=x.shape;d=self.head_dim;cos,sin=positions
        q=self.q_proj(x).view(b,s,self.num_heads,d)
        k=self.k_proj(x).view(b,s,self.num_kv_heads,d)
        v=self.v_proj(x).view(b,s,self.num_kv_heads,d)
        q=norm_rope(q,self.q_norm.weight,cos,sin,self.q_norm.eps).transpose(1,2).contiguous()
        k=norm_rope(k,self.k_norm.weight,cos,sin,self.k_norm.eps).transpose(1,2).contiguous()
        y=causal_attention(q,k,v.transpose(1,2).contiguous(),scale=d**-0.5,optimized=self.options.attention_optimized)
        return self.o_proj(y.transpose(1,2).contiguous().view(b,s,self.num_heads*d))

class OptimizedMLP(TritonMLP):
    def forward(self,x):
        if not self.gate_proj.options.mlp_fusion:return super().forward(x)
        y=gate_up_swiglu(x,self.gate_proj.weight,self.up_proj.weight,optimized=self.gate_proj.options.linear_optimized)
        return self.down_proj(y)

class OptimizedTritonQwen3ForCausalLM(TritonQwen3ForCausalLM):
    def __init__(self,config,options=None):
        options=options or OptimizationOptions();super().__init__(config,options)
        # Retain baseline initialization/weight values. Parameters are assigned,
        # not numerically transformed; official checkpoint loading remains strict.
        for layer in self.model.layers:
            attention=OptimizedAttention(config,options)
            attention.load_state_dict(layer.self_attn.state_dict(),strict=True,assign=True)
            mlp=OptimizedMLP(config,options)
            mlp.load_state_dict(layer.mlp.state_dict(),strict=True,assign=True)
            layer.self_attn=attention;layer.mlp=mlp

VARIANTS=('triton_base','triton_optimized','ablate_attention','ablate_norm_rope','ablate_mlp','ablate_linear','ablate_loss')

def options_for_variant(name):
    if name=='triton_base':return ComputeOptions()
    fields={'ablate_attention':'attention_optimized','ablate_norm_rope':'norm_rope_fusion','ablate_mlp':'mlp_fusion','ablate_linear':'linear_optimized','ablate_loss':'loss_optimized'}
    if name=='triton_optimized':return OptimizationOptions()
    if name in fields:return replace(OptimizationOptions(),**{fields[name]:False})
    raise ValueError('Unknown Triton variant: '+str(name))

def model_for_variant(config,name):
    options=options_for_variant(name)
    if name=='triton_base':return TritonQwen3ForCausalLM(config,options)
    return OptimizedTritonQwen3ForCausalLM(config,options)
