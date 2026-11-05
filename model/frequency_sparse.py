"""Opt-in research cache; the default model/static_decode.py is untouched.

FASA-inspired, per-query-head calibrated pairs, full GPU KV retained.
No sparse training/backward, padding, beams, eviction, offload or paged cache.
"""
import torch
from model.static_decode import StaticDecodeCache, StaticDecodeRunner
from ops.triton.frequency_sparse import pack_proxy, proxy_scores, indexed_attention, validate_pairs


class FrequencySparseCache(StaticDecodeCache):
    def __init__(self, config, batch_size, capacity, *, device, dtype, pairs, budget=512):
        super().__init__(config,batch_size,capacity,device=device,dtype=dtype)
        if config.head_dim!=128 or not 1<=budget<=capacity:
            raise ValueError('Experimental sparse cache requires D=128 and budget<=capacity')
        if len(pairs)!=config.num_hidden_layers:
            raise ValueError('Pairs must cover all layers')
        self.pairs=[p.to(device=device,dtype=torch.int32).contiguous() for p in pairs]
        for p in self.pairs:
            validate_pairs(p,config.num_attention_heads)
        self.budget=budget
        self.proxy=[torch.empty((batch_size,config.num_attention_heads,capacity,2*p.shape[-1]),
                               device=device,dtype=dtype) for p in self.pairs]

    def pack_prefill(self):
        for layer,pairs in enumerate(self.pairs):
            pack_proxy(self.key_cache[layer],pairs,self.positions,self.proxy[layer])

    def attend_decode(self,q,layer_idx,scale):
        if q.shape[2]!=1:
            raise ValueError('Single-query decode only')
        # Direct K/V stores in hf_attention bypass cache.update; read those
        # freshly written slots here. Includes proxy-update cost in every step.
        pack_proxy(self.key_cache[layer_idx],self.pairs[layer_idx],self.positions,self.proxy[layer_idx])
        q=q[:,:,0]
        scores=proxy_scores(q,self.proxy[layer_idx],self.pairs[layer_idx],self.used)
        indices=scores.topk(self.budget,dim=-1,sorted=False).indices
        out=indexed_attention(q,self.key_cache[layer_idx],self.value_cache[layer_idx],
                              indices,self.used,scale=scale)
        return out.unsqueeze(1)

    @property
    def proxy_nbytes(self):
        return sum(t.numel()*t.element_size() for t in self.proxy)


class FrequencySparseRunner(StaticDecodeRunner):
    def __init__(self,model,batch_size,capacity,*,pairs,budget=512):
        super().__init__(model,batch_size,capacity)
        p=next(model.parameters())
        # Constructor-only replacement; no global patch or default change.
        self.cache=FrequencySparseCache(model.config,batch_size,capacity,
            device=p.device,dtype=p.dtype,pairs=pairs,budget=budget)

    def prefill(self,tokens):
        if tokens.shape[1]<2:
            raise ValueError('Sparse experiment requires prefill length >=2')
        out=super().prefill(tokens)
        self.cache.pack_prefill()
        return out
