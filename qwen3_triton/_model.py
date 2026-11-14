"""Trainable Qwen3 graph with explicit Triton model arithmetic.

Dense, unpadded causal training; no KV cache, sliding window, attention dropout,
RoPE scaling or attention bias yet. Unsupported features fail explicitly.
Uses official Qwen3 parameter names for direct state-dict parity checks.
"""
from dataclasses import dataclass
import torch
from torch import nn
import triton
import triton.language as tl
from transformers.modeling_outputs import CausalLMOutputWithPast
from .kernels.linear import linear
from .kernels.embedding import embedding
from .kernels.residual import residual_add
from .kernels.norm import rms_norm
from .kernels.rotary import rotary_tables,apply_rotary
from .kernels.activation import swiglu
from .kernels.loss import causal_cross_entropy
from .kernels.attention import causal_attention,cast_tensor

@dataclass(frozen=True)
class ComputeOptions:
    """Intermediate tiling switches; not the final fusion/ablation suite."""
    linear_optimized: bool=False
    attention_optimized: bool=False
    loss_optimized: bool=False

@triton.jit
def _positions(P,N:tl.constexpr,S:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(P+i,i%S,i<N)

class TritonLinear(nn.Module):
    def __init__(self,in_features,out_features,std,options):
        super().__init__();self.weight=nn.Parameter(torch.empty(out_features,in_features));nn.init.normal_(self.weight,std=std);self.options=options
    def forward(self,x):return linear(x,self.weight,optimized=self.options.linear_optimized)

class TritonEmbedding(nn.Module):
    def __init__(self,vocab,hidden,std):
        super().__init__();self.weight=nn.Parameter(torch.empty(vocab,hidden));nn.init.normal_(self.weight,std=std)
    def forward(self,ids):return embedding(ids,self.weight)

class TritonNorm(nn.Module):
    def __init__(self,hidden,eps):
        super().__init__();self.weight=nn.Parameter(torch.ones(hidden));self.eps=eps
    def forward(self,x):return rms_norm(x,self.weight,self.eps)

class TritonRotary(nn.Module):
    def __init__(self,config):
        super().__init__()
        # Static frequency construction on CPU is configuration materialization.
        # Keep actual CPU values even when parameter construction uses meta.
        freq=1./(config.rope_theta**(torch.arange(0,config.head_dim,2,device='cpu',dtype=torch.float32)/config.head_dim))
        self.register_buffer('inv_freq',freq,persistent=False)
    def forward(self,positions,dtype):
        freq=self.inv_freq if self.inv_freq.dtype==torch.float32 else cast_tensor(self.inv_freq,torch.float32)
        return rotary_tables(positions,freq,dtype)

class TritonAttention(nn.Module):
    def __init__(self,c,options):
        super().__init__();self.options=options;self.head_dim=c.head_dim;self.num_heads=c.num_attention_heads;self.num_kv_heads=c.num_key_value_heads
        h=c.hidden_size;std=c.initializer_range
        self.q_proj=TritonLinear(h,self.num_heads*self.head_dim,std,options)
        self.k_proj=TritonLinear(h,self.num_kv_heads*self.head_dim,std,options)
        self.v_proj=TritonLinear(h,self.num_kv_heads*self.head_dim,std,options)
        self.o_proj=TritonLinear(self.num_heads*self.head_dim,h,std,options)
        self.q_norm=TritonNorm(self.head_dim,c.rms_norm_eps);self.k_norm=TritonNorm(self.head_dim,c.rms_norm_eps)
    def forward(self,x,positions):
        b,s,_=x.shape;d=self.head_dim;cos,sin=positions
        q=self.q_norm(self.q_proj(x).view(b,s,self.num_heads,d))
        k=self.k_norm(self.k_proj(x).view(b,s,self.num_kv_heads,d))
        v=self.v_proj(x).view(b,s,self.num_kv_heads,d)
        q=apply_rotary(q,cos,sin).transpose(1,2).contiguous()
        k=apply_rotary(k,cos,sin).transpose(1,2).contiguous();v=v.transpose(1,2).contiguous()
        y=causal_attention(q,k,v,scale=d**-0.5,optimized=self.options.attention_optimized)
        return self.o_proj(y.transpose(1,2).contiguous().view(b,s,self.num_heads*d))

class TritonMLP(nn.Module):
    def __init__(self,c,options):
        super().__init__();h=c.hidden_size;i=c.intermediate_size;std=c.initializer_range
        self.gate_proj=TritonLinear(h,i,std,options);self.up_proj=TritonLinear(h,i,std,options);self.down_proj=TritonLinear(i,h,std,options)
    def forward(self,x):return self.down_proj(swiglu(self.gate_proj(x),self.up_proj(x)))

class TritonDecoderLayer(nn.Module):
    def __init__(self,c,options):
        super().__init__();self.self_attn=TritonAttention(c,options);self.mlp=TritonMLP(c,options)
        self.input_layernorm=TritonNorm(c.hidden_size,c.rms_norm_eps);self.post_attention_layernorm=TritonNorm(c.hidden_size,c.rms_norm_eps)
    def forward(self,x,position_embeddings):
        x=residual_add(x,self.self_attn(self.input_layernorm(x),position_embeddings))
        return residual_add(x,self.mlp(self.post_attention_layernorm(x)))

class TritonQwen3Model(nn.Module):
    def __init__(self,c,options):
        super().__init__();self.embed_tokens=TritonEmbedding(c.vocab_size,c.hidden_size,c.initializer_range)
        self.layers=nn.ModuleList([TritonDecoderLayer(c,options) for _ in range(c.num_hidden_layers)])
        self.norm=TritonNorm(c.hidden_size,c.rms_norm_eps);self.rotary_emb=TritonRotary(c)
    def forward(self,ids,position_ids=None):
        x=self.embed_tokens(ids)
        if position_ids is None:
            position_ids=torch.empty(ids.shape,device=ids.device,dtype=torch.int64)
            _positions[(triton.cdiv(ids.numel(),256),)](position_ids,ids.numel(),ids.shape[1],256)
        if position_ids.shape==(1,ids.shape[1]) and ids.shape[0]!=1:position_ids=position_ids.expand(ids.shape[0],-1).contiguous()
        if position_ids.shape!=ids.shape:raise ValueError('position_ids must match batch and sequence')
        positions=self.rotary_emb(position_ids,x.dtype)
        for layer in self.layers:x=layer(x,positions)
        return self.norm(x)

class TritonQwen3ForCausalLM(nn.Module):
    def __init__(self,config,options=None):
        super().__init__();self.config=config;self.options=options or ComputeOptions()
        if config.model_type!='qwen3':raise ValueError('Official Qwen3 config required')
        if config.hidden_act!='silu' or config.attention_dropout!=0 or getattr(config,'attention_bias',False):raise ValueError('Only SiLU, zero dropout, bias-free Qwen3 supported')
        if getattr(config,'use_sliding_window',False) or getattr(config,'rope_scaling',None):raise ValueError('Sliding window and scaledRoPE are not implemented')
        if config.head_dim%2 or config.num_attention_heads%config.num_key_value_heads:raise ValueError('Even rotary dimension and valid GQA required')
        self.model=TritonQwen3Model(config,self.options);self.lm_head=TritonLinear(config.hidden_size,config.vocab_size,config.initializer_range,self.options)
        self.tie_weights()
    def tie_weights(self):
        if self.config.tie_word_embeddings:self.lm_head.weight=self.model.embed_tokens.weight
    def load_hf_state_dict(self,state,*,assign=False):
        state=dict(state)
        if self.config.tie_word_embeddings and 'lm_head.weight' not in state:state['lm_head.weight']=state['model.embed_tokens.weight']
        if self.config.tie_word_embeddings and not torch.equal(state['lm_head.weight'],state['model.embed_tokens.weight']):raise ValueError('Checkpoint disagrees with tied embedding configuration')
        result=self.load_state_dict(state,strict=True,assign=assign);self.tie_weights();return result
    def forward(self,input_ids,labels=None,*,position_ids=None,use_cache=False,attention_mask=None,past_key_values=None,output_attentions=False,output_hidden_states=False,return_dict=True,logits_to_keep=0):
        if use_cache or past_key_values is not None or attention_mask is not None or output_attentions or output_hidden_states or logits_to_keep:
            raise ValueError('Only dense unpadded causal training without cache/extra outputs is implemented; no fallback')
        if input_ids.ndim!=2 or input_ids.dtype not in (torch.int32,torch.int64):raise ValueError('Integer [B,S] input_ids required')
        hidden=self.model(input_ids,position_ids);logits=self.lm_head(hidden)
        loss=causal_cross_entropy(logits,labels,optimized=self.options.loss_optimized) if labels is not None else None
        if not return_dict:return (loss,logits) if labels is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss,logits=logits)
