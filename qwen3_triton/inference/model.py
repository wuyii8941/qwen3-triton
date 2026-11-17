"""Dense BF16 Qwen3 inference with explicit Triton GPU computation.

Single device, unpadded equal-length batches, contiguous cache, greedy decoding.
Host model/cache objects and CUDA resource management remain Python/PyTorch.
Unknown GPU tensor operations raise through the same strict mode as V14.
"""
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig,GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from .._model import TritonQwen3ForCausalLM
from ..configuration import OptimizationOptions
from ..kernels.embedding import embedding
from ..kernels.math_precision import swiglu
from .kernels import rms_norm,norm_rope
from ..kernels.mlp_precision import gate_up_swiglu
from ..kernels.residual import residual_add
from .dispatch import InferenceRuntimeMode
from ..runtime.memory import raw_empty,zeros
from ..runtime.ops import copy_
from . import kernels


@dataclass
class KVCache:
    """Request-owned [layer][batch,KV-head,capacity,head-dim] storage.

    Only positions below length are readable; reset keeps allocation and discards
    logical contents. A failed append invalidates the cache until explicit reset.
    """
    keys: list
    values: list
    capacity: int
    batch_size: int
    owner: object
    length: int=0
    valid: bool=True

    @property
    def nbytes(self):return sum(t.numel()*t.element_size() for t in self.keys+self.values)

    def reset(self):self.length=0;self.valid=True


class TritonQwen3Inference(nn.Module):
    def __init__(self,config,*,gemv=True,split_decode=False,sliced_decode=False,prefill_tiling=False,reference_accumulation=False):
        super().__init__()
        if config.head_dim!=128:raise ValueError('The qualified Qwen3 attention head dimension is128')
        # Meta construction avoids allocating/randomly initializing another8B.
        with torch.device('meta'):
            core=TritonQwen3ForCausalLM(config,OptimizationOptions())
        self.model=core.model;self.lm_head=core.lm_head;self.config=config
        self.generation_config=GenerationConfig.from_model_config(config)
        self.gemv=gemv;self.split_decode=split_decode;self.sliced_decode=sliced_decode;self.prefill_tiling=prefill_tiling;self.reference_accumulation=reference_accumulation;self.runtime=InferenceRuntimeMode();self._cache_owner=object()
        self.eval()

    def load_hf_state_dict(self,state):
        state=dict(state)
        if self.config.tie_word_embeddings and 'lm_head.weight' not in state:state['lm_head.weight']=state['model.embed_tokens.weight']
        if self.config.tie_word_embeddings and not torch.equal(state['lm_head.weight'],state['model.embed_tokens.weight']):raise ValueError('Tied checkpoint weights disagree')
        self.load_state_dict(state,assign=True,strict=True)
        if self.config.tie_word_embeddings:self.lm_head.weight=self.model.embed_tokens.weight
        if any(p.dtype!=torch.bfloat16 for p in self.parameters()):raise ValueError('BF16 inference checkpoint required')
        return self

    @classmethod
    def from_pretrained(cls,path,*,device='cuda',gemv=True,split_decode=False,sliced_decode=False,prefill_tiling=False,reference_accumulation=False):
        """Load local official safetensors; no HF model/GPU compute dependency."""
        import json
        from safetensors import safe_open
        path=Path(path).resolve();config=AutoConfig.from_pretrained(path,local_files_only=True)
        model=cls(config,gemv=gemv,split_decode=split_decode,sliced_decode=sliced_decode,prefill_tiling=prefill_tiling,reference_accumulation=reference_accumulation)
        if (path/'generation_config.json').is_file():
            model.generation_config=GenerationConfig.from_pretrained(path,local_files_only=True)
        index=path/'model.safetensors.index.json'
        shards=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
        state={}
        for name in shards:
            shard=(path/name).resolve()
            if not shard.is_relative_to(path):raise ValueError('Checkpoint shard escapes model directory')
            with safe_open(shard,framework='pt',device='cpu') as f:
                for key in f.keys():state[key]=f.get_tensor(key)
        model.load_hf_state_dict(state);del state
        # Frequencies are FP32 configuration values; do not dtype-cast them.
        with model.runtime:model.to(device=device)
        return model

    @property
    def device(self):return self.lm_head.weight.device

    def new_cache(self,batch_size,max_length):
        if type(batch_size) is not int or batch_size<1 or type(max_length) is not int or not 1<=max_length<=self.config.max_position_embeddings:
            raise ValueError('Positive batch/capacity within max_position_embeddings required')
        if self.device.type!='cuda':raise ValueError('Load BF16 weights onto a CUDA device first')
        shape=(batch_size,self.config.num_key_value_heads,max_length,self.config.head_dim)
        keys=[raw_empty(shape,device=self.device,dtype=torch.bfloat16) for _ in self.model.layers]
        values=[raw_empty(shape,device=self.device,dtype=torch.bfloat16) for _ in self.model.layers]
        return KVCache(keys,values,max_length,batch_size,self._cache_owner)

    def _forward(self,input_ids,cache,logits_to_keep,*,validate_ids=True):
        if input_ids.ndim!=2 or input_ids.dtype not in (torch.int32,torch.int64) or input_ids.device!=self.device or not input_ids.is_contiguous():
            raise ValueError('Contiguous integer CUDA[B,S] input_ids on model device required')
        b,s=input_ids.shape
        if b<1 or s<1:raise ValueError('Nonempty batch/sequence required')
        if cache.owner is not self._cache_owner or not cache.valid or b!=cache.batch_size or cache.length+s>cache.capacity:
            raise ValueError('Foreign, invalid, mismatched or exhausted KV cache')
        if type(logits_to_keep) is not int or not 0<=logits_to_keep<=s:raise ValueError('logits_to_keep must be0 or within current sequence length')
        # Prevent out-of-bounds embedding loads before launching any cache update.
        if validate_ids and not bool(((input_ids>=0)&(input_ids<self.config.vocab_size)).all()):raise ValueError('Token ID outside vocabulary')
        start=cache.length;pos=kernels.positions(b,s,start,self.device)
        try:
            x=embedding(input_ids,self.model.embed_tokens.weight)
            cos,sin=self.model.rotary_emb(pos,torch.bfloat16)
            for i,layer in enumerate(self.model.layers):
                h=rms_norm(x,layer.input_layernorm.weight,layer.input_layernorm.eps,reference_accumulation=self.reference_accumulation)
                a=layer.self_attn;d=a.head_dim
                q=kernels.linear(h,a.q_proj.weight,gemv=self.gemv,sliced=self.sliced_decode,prefill_tiling=self.prefill_tiling,reference_accumulation=self.reference_accumulation).view(b,s,a.num_heads,d)
                k=kernels.linear(h,a.k_proj.weight,gemv=self.gemv,sliced=self.sliced_decode,prefill_tiling=self.prefill_tiling,reference_accumulation=self.reference_accumulation).view(b,s,a.num_kv_heads,d)
                v=kernels.linear(h,a.v_proj.weight,gemv=self.gemv,sliced=self.sliced_decode,prefill_tiling=self.prefill_tiling,reference_accumulation=self.reference_accumulation).view(b,s,a.num_kv_heads,d)
                q=norm_rope(q,a.q_norm.weight,cos,sin,a.q_norm.eps)
                k=norm_rope(k,a.k_norm.weight,cos,sin,a.k_norm.eps)
                kernels.append(k,v,cache.keys[i],cache.values[i],start)
                y=kernels.attention(q,cache.keys[i],cache.values[i],start,split_decode=self.split_decode).view(b,s,-1)
                x=residual_add(x,kernels.linear(y,a.o_proj.weight,gemv=self.gemv,sliced=self.sliced_decode,prefill_tiling=self.prefill_tiling,reference_accumulation=self.reference_accumulation))
                h=rms_norm(x,layer.post_attention_layernorm.weight,layer.post_attention_layernorm.eps,reference_accumulation=self.reference_accumulation)
                m=layer.mlp
                if (b*s<=8 and self.gemv) or (self.reference_accumulation and b*s==17 and h.shape[-1]==4096 and m.gate_proj.weight.shape[0]==12288):
                    y=swiglu(kernels.linear(h,m.gate_proj.weight,gemv=self.gemv,reference_accumulation=self.reference_accumulation),kernels.linear(h,m.up_proj.weight,gemv=self.gemv,reference_accumulation=self.reference_accumulation))
                elif self.prefill_tiling and b*s==65 and h.shape[-1]==4096 and m.gate_proj.weight.shape[0]==12288:
                    y=kernels.tiled_prefill_gate_up(h,m.gate_proj.weight,m.up_proj.weight)
                else:y=gate_up_swiglu(h,m.gate_proj.weight,m.up_proj.weight,optimized=True)
                x=residual_add(x,kernels.linear(y,m.down_proj.weight,gemv=self.gemv,sliced=self.sliced_decode,prefill_tiling=self.prefill_tiling,reference_accumulation=self.reference_accumulation))
            x=rms_norm(x,self.model.norm.weight,self.model.norm.eps,reference_accumulation=self.reference_accumulation)
            if logits_to_keep:x=x[:,-logits_to_keep:,:].contiguous()
            logits=kernels.linear(x,self.lm_head.weight,gemv=self.gemv,sliced=self.sliced_decode,prefill_tiling=self.prefill_tiling,reference_accumulation=self.reference_accumulation)
            cache.length=start+s
        except BaseException:
            cache.valid=False
            raise
        return CausalLMOutputWithPast(logits=logits,past_key_values=cache)

    @torch.inference_mode()
    def forward(self,input_ids,*,past_key_values=None,use_cache=True,logits_to_keep=1):
        """Append input chunk; cache.length supplies absolute RoPE/causal offsets.

        use_cache=False permits full-sequence recomputation for verification.
        Masked/padded batches and arbitrary position_ids are intentionally absent.
        """
        if not use_cache and past_key_values is not None:raise ValueError('A supplied cache requires use_cache=True')
        with self.runtime:
            cache=past_key_values or self.new_cache(input_ids.shape[0],input_ids.shape[1])
            out=self._forward(input_ids,cache,logits_to_keep)
        if not use_cache:out.past_key_values=None
        return out

    @torch.inference_mode()
    def generate(self,input_ids,*,max_new_tokens=32,eos_token_id=None,pad_token_id=None):
        """Greedy GPU argmax with per-row EOS, padding and request-local cache."""
        if type(max_new_tokens) is not int or max_new_tokens<0:raise ValueError('Nonnegative max_new_tokens required')
        eos=self.generation_config.eos_token_id if eos_token_id is None else eos_token_id
        eos=() if eos is None else ((eos,) if isinstance(eos,int) else tuple(eos))
        if any(type(x) is not int or not 0<=x<self.config.vocab_size for x in eos):raise ValueError('Invalid EOS tokens')
        configured_pad=self.generation_config.pad_token_id
        pad=pad_token_id if pad_token_id is not None else (configured_pad if configured_pad is not None else (eos[0] if eos else 0))
        if type(pad) is not int or not 0<=pad<self.config.vocab_size:raise ValueError('Invalid padding token')
        with self.runtime:
            if max_new_tokens==0:
                # Same validation as forward, without modifying a cache.
                if input_ids.ndim!=2 or input_ids.device!=self.device or input_ids.dtype not in (torch.int32,torch.int64) or min(input_ids.shape)<1:raise ValueError('Nonempty integer CUDA[B,S] required')
                if not bool(((input_ids>=0)&(input_ids<self.config.vocab_size)).all()):raise ValueError('Token ID outside vocabulary')
                return input_ids.clone()
            b,s=input_ids.shape
            cache=self.new_cache(b,s+max_new_tokens)
            output=raw_empty((b,s+max_new_tokens),device=self.device,dtype=torch.int64)
            copy_(output[:,:s],input_ids)
            done=zeros((b,),device=self.device,dtype=torch.bool)
            # Validate once: future greedy tokens are in-vocabulary by construction.
            if not torch.cuda.is_current_stream_capturing():
                if not bool(((input_ids>=0)&(input_ids<self.config.vocab_size)).all()):raise ValueError('Token ID outside vocabulary')
            current=input_ids
            for step in range(max_new_tokens):
                logits=self._forward(current,cache,1,validate_ids=False).logits[:,0,:]
                current=raw_empty((b,1),device=self.device,dtype=torch.int64)
                kernels.select(logits,output,current,done,s+step,eos,pad)
                if eos and bool(done.all()):break
            return output[:,:s+step+1].contiguous()


class CapturedGreedyRequest:
    """Fixed dense shape and fixed token budget; graph owns buffers, no earlyEOS.

    One object is single-stream/non-concurrent. Returned tokens alias request
    storage until next replay; clone for retention. Capture validates prompts
    before replay. CUDA graph is a host launch service, GPU math remains Triton.
    """
    @torch.inference_mode()
    def __init__(self,model,input_ids,*,max_new_tokens):
        if type(max_new_tokens) is not int or max_new_tokens<1:raise ValueError('Positive fixed token budget required')
        self.model=model;self.shape=tuple(input_ids.shape);self.busy=False
        with model.runtime:
            self.input=input_ids.clone()
            if not bool(((self.input>=0)&(self.input<model.config.vocab_size)).all()):raise ValueError('Token ID outside vocabulary')
        stream=torch.cuda.Stream(device=model.device)
        stream.wait_stream(torch.cuda.current_stream(model.device))
        with torch.cuda.stream(stream):
            for _ in range(2):model.generate(self.input,max_new_tokens=max_new_tokens,eos_token_id=[],pad_token_id=0)
        torch.cuda.current_stream(model.device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            self.output=model.generate(self.input,max_new_tokens=max_new_tokens,eos_token_id=[],pad_token_id=0)

    @torch.inference_mode()
    def __call__(self,input_ids):
        if self.busy:raise RuntimeError('Concurrent graph request reuse is unsupported')
        if tuple(input_ids.shape)!=self.shape or input_ids.device!=self.model.device or input_ids.dtype!=self.input.dtype:raise ValueError('Captured input shape/device/dtype differs')
        self.busy=True
        try:
            with self.model.runtime:
                if not bool(((input_ids>=0)&(input_ids<self.model.config.vocab_size)).all()):raise ValueError('Token ID outside vocabulary')
                copy_(self.input,input_ids)
            self.graph.replay()
            return self.output
        finally:self.busy=False
