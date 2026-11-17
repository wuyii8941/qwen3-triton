"""Training components with reference-order Triton arithmetic."""
from ._model import TritonAttention, TritonMLP, TritonNorm, TritonDecoderLayer, TritonQwen3Model
from .kernels.rotary import apply_rotary
from .kernels.residual import residual_add
from .kernels.attention_precision import candidate
from .kernels.math_precision import rms_norm, norm_rope, swiglu
from .kernels.mlp_precision import gate_up_swiglu

class HFOrderNorm(TritonNorm):
    def forward(self, x): return rms_norm(x, self.weight, self.eps)

class Attention(TritonAttention):
    def __init__(self, config, options):
        super().__init__(config, options)
        self.precision_options = dict(options.precision_options)
        self.q_norm = HFOrderNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = HFOrderNorm(self.head_dim, config.rms_norm_eps)
    def forward(self, x, positions):
        b,s,_=x.shape; d=self.head_dim; cos,sin=positions
        q=self.q_proj(x).view(b,s,self.num_heads,d)
        k=self.k_proj(x).view(b,s,self.num_kv_heads,d)
        v=self.v_proj(x).view(b,s,self.num_kv_heads,d)
        if self.options.norm_rope_fusion:
            q=norm_rope(q,self.q_norm.weight,cos,sin,self.q_norm.eps)
            k=norm_rope(k,self.k_norm.weight,cos,sin,self.k_norm.eps)
        else:
            q=apply_rotary(self.q_norm(q),cos,sin)
            k=apply_rotary(self.k_norm(k),cos,sin)
        q=q.transpose(1,2).contiguous();k=k.transpose(1,2).contiguous();v=v.transpose(1,2).contiguous()
        y=candidate(q,k,v,fusion=False,lane_sum=True,round_kv=True,reciprocal=1,stages=2 if self.options.attention_optimized else 1,**self.precision_options)
        return self.o_proj(y.transpose(1,2).contiguous().view(b,s,self.num_heads*d))

class MLP(TritonMLP):
    def forward(self,x):
        opts=self.gate_proj.options
        if opts.mlp_fusion:
            y=gate_up_swiglu(x,self.gate_proj.weight,self.up_proj.weight,optimized=opts.linear_optimized)
        else:
            y=swiglu(self.gate_proj(x),self.up_proj(x))
        return self.down_proj(y)

class DecoderLayer(TritonDecoderLayer):
    def __init__(self,c,options):
        import torch
        torch.nn.Module.__init__(self)
        self.self_attn=Attention(c,options);self.mlp=MLP(c,options)
        self.input_layernorm=HFOrderNorm(c.hidden_size,c.rms_norm_eps)
        self.post_attention_layernorm=HFOrderNorm(c.hidden_size,c.rms_norm_eps)

class Backbone(TritonQwen3Model):
    decoder_class=DecoderLayer
    def __init__(self,c,options):
        super().__init__(c,options)
        self.norm=HFOrderNorm(c.hidden_size,c.rms_norm_eps)
