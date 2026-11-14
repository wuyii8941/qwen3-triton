"""Unqualified full8B candidate with explicit selectable native-order attention controls.

All model arithmetic remains explicit Triton. The attention optimization
changes forward software-pipeline stages, preserving reduction/cast choices.
Optional math controls are separate from the five performance ablations.
"""
from dataclasses import fields
from qwen3_triton._model import TritonAttention,TritonMLP,TritonNorm
from qwen3_triton._model_optimized import OptimizationOptions,OptimizedTritonQwen3ForCausalLM,options_for_variant
from qwen3_triton.kernels.rotary import apply_rotary
from qwen3_triton.kernels.attention import causal_attention as old_attention
from qwen3_triton.kernels.norm_rope import norm_rope as old_norm_rope
from qwen3_triton.kernels.activation import swiglu as old_swiglu
from qwen3_triton.kernels.mlp import gate_up_swiglu as old_gate_up
from .kernels.attention_precision import candidate
from .kernels.math_precision import rms_norm,norm_rope,swiglu
from .kernels.mlp_precision import gate_up_swiglu

class HFOrderNorm(TritonNorm):
    def forward(self,x):return rms_norm(x,self.weight,self.eps)

class Precision2Attention(TritonAttention):
    def __init__(self,c,options,hf_norms,native_attention,precision_options):
        super().__init__(c,options);self.hf_norms=hf_norms;self.native_attention=native_attention;self.precision_options=dict(precision_options)
    def forward(self,x,positions):
        b,s,_=x.shape;d=self.head_dim;cos,sin=positions
        q=self.q_proj(x).view(b,s,self.num_heads,d);k=self.k_proj(x).view(b,s,self.num_kv_heads,d)
        v=self.v_proj(x).view(b,s,self.num_kv_heads,d)
        if self.options.norm_rope_fusion:
            fn=norm_rope if self.hf_norms else old_norm_rope
            q=fn(q,self.q_norm.weight,cos,sin,self.q_norm.eps);k=fn(k,self.k_norm.weight,cos,sin,self.k_norm.eps)
        else:
            q=apply_rotary(self.q_norm(q),cos,sin);k=apply_rotary(self.k_norm(k),cos,sin)
        q=q.transpose(1,2).contiguous();k=k.transpose(1,2).contiguous();v=v.transpose(1,2).contiguous()
        if self.native_attention:
            y=candidate(q,k,v,fusion=False,lane_sum=True,round_kv=True,reciprocal=1,stages=2 if self.options.attention_optimized else 1,**self.precision_options)
        else:y=old_attention(q,k,v,optimized=self.options.attention_optimized)
        return self.o_proj(y.transpose(1,2).contiguous().view(b,s,self.num_heads*d))

class Precision2MLP(TritonMLP):
    def __init__(self,c,options,hf_activation):super().__init__(c,options);self.hf_activation=hf_activation
    def forward(self,x):
        opts=self.gate_proj.options
        if opts.mlp_fusion:
            fn=gate_up_swiglu if self.hf_activation else old_gate_up
            y=fn(x,self.gate_proj.weight,self.up_proj.weight,optimized=opts.linear_optimized)
        else:
            fn=swiglu if self.hf_activation else old_swiglu
            y=fn(self.gate_proj(x),self.up_proj(x))
        return self.down_proj(y)

class PrecisionQwen3ForCausalLM(OptimizedTritonQwen3ForCausalLM):
    def __init__(self,config,options=None,*,precision_options=None,hf_norms=True,hf_activation=True,native_attention=True):
        if precision_options is None:precision_options=dict(xor_sum=True,lse_fma=True,native_delta=True,native_dq=True,sum_fma=True)
        if set(precision_options)!={'xor_sum','lse_fma','native_delta','native_dq','sum_fma'} or any(type(v) is not bool for v in precision_options.values()):raise ValueError('Explicit fiveboolean precision options required')
        assert config.num_hidden_layers==36 and config.hidden_size==4096 and config.head_dim==128
        if options is None:options=OptimizationOptions(**{f.name:False for f in fields(OptimizationOptions)})
        elif not isinstance(options,OptimizationOptions):options=OptimizationOptions(**{f.name:getattr(options,f.name,False) for f in fields(OptimizationOptions)})
        super().__init__(config,options)
        for layer in self.model.layers:
            attention=Precision2Attention(config,options,hf_norms,native_attention,precision_options)
            attention.load_state_dict(layer.self_attn.state_dict(),strict=True,assign=True)
            mlp=Precision2MLP(config,options,hf_activation);mlp.load_state_dict(layer.mlp.state_dict(),strict=True,assign=True)
            layer.self_attn=attention;layer.mlp=mlp
        if hf_norms:
            for name,module in list(self.named_modules()):
                if isinstance(module,TritonNorm):
                    parent,attribute=name.rsplit('.',1);replacement=HFOrderNorm(module.weight.numel(),module.eps)
                    replacement.weight=module.weight;setattr(self.get_submodule(parent),attribute,replacement)

def model_for_variant(config,name,*,precision_options=None):return PrecisionQwen3ForCausalLM(config,options_for_variant(name),precision_options=precision_options)
