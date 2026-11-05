"""Qwen3 4.51 attention epilogue ablation, preserving HF cache/mask behavior.

We keep the projections and output projection as PyTorch Linear. This adapts
the installed HF Qwen3Attention.forward, not the attention math/backend.
"""
import torch
from transformers.models.qwen3 import modeling_qwen3 as hf
from ops.triton.hf_qk_norm_rope import _launch
from model.static_decode import StaticDecodeCache


def make_attention_forward(*,fuse_qk,compact_gqa,packed_qkv=False,direct_k_cache=False,
                           direct_v_cache=False,rope_fn=None):
    def forward(self,hidden_states,position_embeddings,attention_mask=None,
                past_key_value=None,cache_position=None,**kwargs):
        input_shape=hidden_states.shape[:-1]
        head_shape=(*input_shape,-1,self.head_dim)
        if packed_qkv:
            weight,bias=self._packed_qkv
            packed=torch.nn.functional.linear(hidden_states,weight,bias)
            sizes=(self.q_proj.out_features,self.k_proj.out_features,self.v_proj.out_features)
            q,k,v=(p.view(head_shape) for p in packed.split(sizes,dim=-1))
            v=v.transpose(1,2)
        else:
            q=self.q_proj(hidden_states).view(head_shape)
            k=self.k_proj(hidden_states).view(head_shape)
            v=self.v_proj(hidden_states).view(head_shape).transpose(1,2)
        static=isinstance(past_key_value,StaticDecodeCache)
        direct=static and direct_k_cache
        cos,sin=position_embeddings
        if fuse_qk:
            q,k=_launch(q,k,self.q_norm.weight,self.k_norm.weight,
                        cos.contiguous(),sin.contiguous(),self.q_norm.variance_epsilon,
                        k_cache=past_key_value.key_cache[self.layer_idx] if direct else None,
                        positions=cache_position if direct else None,
                        v=v if direct and direct_v_cache else None,
                        v_cache=past_key_value.value_cache[self.layer_idx] if direct and direct_v_cache else None)
        else:
            q=self.q_norm(q).transpose(1,2)
            k=self.k_norm(k).transpose(1,2)
            q,k=(hf.apply_rotary_pos_emb if rope_fn is None else rope_fn)(q,k,cos,sin)
        if direct:
            if not direct_v_cache:
                past_key_value.write_value(v,self.layer_idx)
        elif past_key_value is not None:
            k,v=past_key_value.update(k,v,self.layer_idx,
                {'sin':sin,'cos':cos,'cache_position':cache_position})
        if static and q.shape[2]==1:
            output=past_key_value.attend_decode(q,self.layer_idx,self.scaling)
            weights=None
        elif (compact_gqa and self.config._attn_implementation=='sdpa'
                and not kwargs.get('output_attentions',False)):
            mask=attention_mask
            if mask is not None and mask.ndim==4:
                mask=mask[:,:,:,:k.shape[-2]]
            causal=kwargs.get('is_causal')
            if causal is None:
                causal=q.shape[2]>1 and mask is None
            # PyTorch 2.5's native GQA avoids HF's expanded-and-copied K/V.
            # Forced Flash in benchmarks; no hidden custom-attention fallback.
            output=torch.nn.functional.scaled_dot_product_attention(q,k,v,
                attn_mask=mask,dropout_p=0.,scale=self.scaling,is_causal=causal,enable_gqa=True)
            output=output.transpose(1,2).contiguous()
            weights=None
        else:
            interface=hf.eager_attention_forward
            impl=self.config._attn_implementation
            if impl!='eager' and not (impl=='sdpa' and kwargs.get('output_attentions',False)):
                interface=hf.ALL_ATTENTION_FUNCTIONS[impl]
            output,weights=interface(self,q,k,v,attention_mask,dropout=0.,scaling=self.scaling,
                                     sliding_window=self.sliding_window,**kwargs)
        return self.o_proj(output.reshape(*input_shape,-1).contiguous()),weights
    return forward
