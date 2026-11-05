"""Experimental, unpadded, equal-length Qwen3 decode on fixed GPU buffers.

Pinned to torch 2.5.1 / Transformers 4.51.0. The private Flash varlen op is
intentional: public SDPA cannot express a GPU-resident valid prefix of a
capacity-sized cache without an attention mask. This is NOT a serving cache;
no padding, beams, chunked prefill, sliding window, or concurrent requests.
"""
import torch
from transformers.cache_utils import Cache


class StaticDecodeCache(Cache):
    def __init__(self, config, batch_size, capacity, *, device, dtype):
        super().__init__()
        if torch.__version__.split('+')[0] != '2.5.1':
            raise ValueError('Private Flash API validated only on torch 2.5.1')
        if min(batch_size, capacity) < 1:
            raise ValueError('Positive batch and capacity required')
        self.batch_size, self.capacity = batch_size, capacity
        self.heads, self.dim = config.num_key_value_heads, config.head_dim
        shape = (batch_size, capacity, self.heads, self.dim)
        # BSHD permits flattening batch/capacity without copying the cache.
        self.key_cache = [torch.empty(shape, device=device, dtype=dtype)
                          for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.empty_like(k) for k in self.key_cache]
        self.cu_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
        self.cu_k = self.cu_q * capacity
        self.used = torch.zeros(batch_size, device=device, dtype=torch.int32)
        self.decode_position = torch.zeros(1, device=device, dtype=torch.long)
        self.positions = self.decode_position
        self.start = self.count = self.end = 0

    def prepare(self, start, count):
        """Host-side checked preparation, always OUTSIDE graph capture/replay."""
        if start < 0 or count < 1 or start + count > self.capacity:
            raise ValueError('KV cache capacity exceeded or invalid position')
        if count > 1 and start != 0:
            raise ValueError('Only initial prefill followed by one-token decode is supported')
        self.start, self.count, self.end = start, count, start + count
        if count == 1:
            self.decode_position.fill_(start)
            self.positions = self.decode_position
        else:
            self.positions = torch.arange(count, device=self.used.device)
        self.used.fill_(self.end)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Inputs BHSD, storage BSHD. No cat or capacity-sized contiguous copy.
        self.key_cache[layer_idx].index_copy_(1, self.positions, key_states.transpose(1, 2))
        self.write_value(value_states, layer_idx)
        return key_states, value_states  # Prefill attends only to its input.

    def write_value(self, value_states, layer_idx):
        self.value_cache[layer_idx].index_copy_(1, self.positions, value_states.transpose(1, 2))

    def attend_decode(self, q, layer_idx, scale):
        b, h, t, d = q.shape
        if t != 1:
            raise ValueError('Decode requires query length 1')
        # seqused_k is honored ONLY in the varlen entry point (cu_q/cu_k set).
        out = torch.ops.aten._flash_attention_forward(
            q.reshape(b, h, d),
            self.key_cache[layer_idx].view(b * self.capacity, self.heads, d),
            self.value_cache[layer_idx].view(b * self.capacity, self.heads, d),
            self.cu_q, self.cu_k, 1, self.capacity, 0., False, False,
            scale=scale, seqused_k=self.used)[0]
        return out.view(b, 1, h, d)

    def get_seq_length(self, layer_idx=0):
        return self.start

    def get_max_cache_shape(self):
        return self.capacity

    def reorder_cache(self, beam_idx):
        raise NotImplementedError('Beam search is outside the static decode contract')

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.key_cache + self.value_cache)


def static_mask_or_original(original):
    def mask(self, attention_mask, input_tensor, cache_position, past_key_values,
             output_attentions=False):
        if not isinstance(past_key_values, StaticDecodeCache):
            return original(attention_mask, input_tensor, cache_position,
                            past_key_values, output_attentions)
        cache = past_key_values
        if attention_mask is not None or output_attentions:
            raise ValueError('Static decode supports no padding/mask/attention outputs')
        if cache_position is not cache.positions or input_tensor.shape[:2] != (cache.batch_size, cache.count):
            raise ValueError('Prepare the static cache and pass its positions explicitly')
        # Prefill uses causal SDPA; decode uses Flash seqused_k, NOT an unmasked
        # public SDPA over the unused tail of the cache.
        return None
    mask._supports_static_decode = True
    return mask


class StaticDecodeRunner:
    """Use within qwen3_triton_inference(..., static_decode=True).

    CUDA Graph captures one forward including last-token LM head. Copies,
    position/length updates, and greedy argmax remain outside the graph and
    must be included in end-to-end timing. Returned logits alias graph output;
    clone them if keeping them across subsequent replay calls.
    """
    def __init__(self, model, batch_size, capacity):
        self.model = model
        p = next(model.parameters())
        self.cache = StaticDecodeCache(model.config, batch_size, capacity,
                                       device=p.device, dtype=p.dtype)
        self.token = torch.empty((batch_size, 1), device=p.device, dtype=torch.long)
        self.length = 0
        self.graph = None
        self.graph_logits = None
        self.packed_refs = ()
        self.quantized_refs = ()

    def _check(self, tokens, *, decode=False):
        if torch.is_grad_enabled() or self.model.training:
            raise ValueError('eval + inference_mode required')
        if not getattr(self.model.model._update_causal_mask, '_supports_static_decode', False):
            raise ValueError('Use the static_decode=True inference context')
        if tokens.ndim != 2 or tokens.shape[0] != self.token.shape[0]:
            raise ValueError('Fixed batch size and [B,S] tokens required')
        if tokens.dtype != torch.long or tokens.device != self.token.device:
            raise ValueError('CUDA int64 tokens on the model device required')
        if decode and tokens.shape[1] != 1:
            raise ValueError('Decode requires exactly one input token')

    def _forward(self, tokens):
        return self.model(tokens, past_key_values=self.cache, use_cache=True,
                          cache_position=self.cache.positions, logits_to_keep=1).logits

    def prefill(self, tokens):
        self._check(tokens)
        self.cache.prepare(0, tokens.shape[1])
        logits = self._forward(tokens)
        self.length = tokens.shape[1]
        return logits

    def capture(self, sample_token):
        self._check(sample_token, decode=True)
        if self.length < 1 or self.graph is not None:
            raise ValueError('Prefill first; capture only once per runner')
        self.cache.prepare(self.length, 1)
        self.token.copy_(sample_token)
        # Warmup/capture overwrite only the next slot, never the valid prefix.
        stream = torch.cuda.Stream(device=self.token.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward(self.token)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = self._forward(self.token)
        self.graph, self.graph_logits = graph, logits
        # Graphs reference raw addresses: explicitly own packed weights even
        # when the scoped forward patch is later removed.
        self.packed_refs = tuple(m._packed_qkv for m in self.model.modules()
                                 if hasattr(m, '_packed_qkv'))
        self.quantized_refs = tuple(m._int8_gate_up for m in self.model.modules()
                                    if hasattr(m, '_int8_gate_up'))
        self.quantized_refs += tuple(m._int8_lm_head for m in self.model.modules()
                                     if hasattr(m,'_int8_lm_head'))

    def decode(self, token, *, replay=True):
        self._check(token, decode=True)
        if self.length < 1:
            raise ValueError('Prefill before decode')
        if replay and self.graph is None:
            raise ValueError('Capture before graph replay')
        self.cache.prepare(self.length, 1)
        if replay:
            self.token.copy_(token)
            self.graph.replay()
            logits = self.graph_logits
        else:
            logits = self._forward(token)
        self.length += 1
        return logits
