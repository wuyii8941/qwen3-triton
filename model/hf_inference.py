"""Scoped, inference-only Transformers 4.51 Qwen3 operator replacement.

RoPE replacement is module-global while the context is active: use in a
single-threaded experiment, not a concurrent serving process.
"""
from contextlib import contextmanager
from types import MethodType

import torch
import transformers
import triton
import triton.language as tl
from transformers.models.qwen3 import modeling_qwen3 as hf
from ops.triton.add_rms_norm import add_rms_norm_triton
from ops.triton.decode_gate_up import decode_gate_up_triton
from ops.triton.rowwise_tuning import _rms_rows
from model.hf_attention import make_attention_forward
from model.static_decode import static_mask_or_original
from ops.triton.gemm_tuning import gate_up_tuned
from ops.triton.weight_only_gemv import quantize_groupwise, gate_up_int8
from ops.triton.linear_decode import linear_decode
from ops import compile_ops


@triton.jit
def _norm(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + row*N + d, d < N, 0).to(tl.float32)
    w = tl.load(W + d, d < N, 0).to(tl.float32)
    z = x * tl.rsqrt(tl.sum(x*x, 0)/N + EPS)
    # HF casts normalized activation before multiplying the learned weight.
    z = z.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + row*N + d, z*w, d < N)


@triton.jit
def _activation(G, U, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G+i, i < N, 0).to(tl.float32)
    u = tl.load(U+i, i < N, 0).to(tl.float32)
    s = (g/(1+tl.exp(-g))).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+i, s*u, i < N)


@triton.jit
def _rotary(X, C, S, Y, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
            XB: tl.constexpr, XH: tl.constexpr, XT: tl.constexpr, XD: tl.constexpr,
            TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    # Flat output tiles handle several head rows per program and accept
    # noncontiguous HF Q/K directly without transpose-contiguous copies.
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    d = i % D
    t = i // D % T
    h = i // (D*T) % H
    b = i // (D*T*H)
    paired = tl.where(d < D//2, d+D//2, d-D//2)
    base = b*XB+h*XH+t*XT
    x = tl.load(X+base+d*XD, i < TOTAL, 0).to(tl.float32)
    xp = tl.load(X+base+paired*XD, i < TOTAL, 0).to(tl.float32)
    c = tl.load(C+(b*T+t)*D+d, i < TOTAL, 0).to(tl.float32)
    s = tl.load(S+(b*T+t)*D+d, i < TOTAL, 0).to(tl.float32)
    # Match the two eager low-precision products before addition.
    left = (x*c).to(Y.dtype.element_ty).to(tl.float32)
    right = (xp*tl.where(d < D//2, -1., 1.)*s).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+i, left+right, i < TOTAL)


def _norm_impl(x, weight, eps):
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.shape[-1]
    _norm[(x.numel()//n,)](x, weight, y, n, eps,
                         triton.next_power_of_2(n), enable_fp_fusion=False)
    return y


def _norm_forward(self, x):
    if torch.compiler.is_compiling():
        return compile_ops.norm(x, self.weight, self.variance_epsilon)
    return _norm_impl(x, self.weight, self.variance_epsilon)


def _norm_multirow_forward(self,x):
    # Narrow, opt-in A6000 heuristic from the D=128 launch sweep. Leave hidden
    # norms and decode's few head rows alone: their measured gains were tiny.
    m=x.numel()//x.shape[-1]
    if x.shape[-1]==128 and m>=2048:
        # The context already establishes the HF/CUDA/inference contract. Avoid
        # repeating the public experimental wrapper's checks at every layer.
        rows=16 if m>=16384 else 4
        x=x.contiguous()
        y=torch.empty_like(x)
        _rms_rows[(triton.cdiv(m,rows),)](x,self.weight,y,m,128,self.variance_epsilon,
            rows,128,num_warps=4,enable_fp_fusion=False)
        return y
    return _norm_forward(self,x)


def _activation_impl(gate, up):
    y = torch.empty_like(gate)
    _activation[(triton.cdiv(y.numel(), 512),)](gate, up, y, y.numel(), 512,
                                               enable_fp_fusion=False)
    return y


def _mlp_forward(self, x):
    gate, up = self.gate_proj(x), self.up_proj(x)
    y = (compile_ops.activation(gate, up) if torch.compiler.is_compiling()
         else _activation_impl(gate, up))
    return self.down_proj(y)


def _mlp_decode_forward(self, x):
    # Measured A6000 specialization: never apply GEMV to prefill / larger M.
    if x.ndim == 3 and x.shape[1] == 1 and 1 <= x.shape[0] <= 2 and x.shape[-1] == 2048:
        y = decode_gate_up_triton(x.reshape(-1,2048).contiguous(),
                                 self.gate_proj.weight,self.up_proj.weight,block_n=1)
        return self.down_proj(y.reshape(x.shape[0],1,-1))
    return _mlp_forward(self,x)


def make_mlp_forward(*, decode_mlp, prefill_mlp, decode_int8=False,decode_int8_impl='naive',
                     decode_int8_max_batch=2):
    def forward(self, x):
        if decode_int8 and x.ndim == 3 and x.shape[1] == 1 and 1 <= x.shape[0] <= decode_int8_max_batch and x.shape[-1] == 2048:
            gate,up=self._int8_gate_up
            bn=4 if decode_int8_impl=='grouped' and x.shape[0]==2 else 1
            y=gate_up_int8(x.reshape(-1,2048).contiguous(),gate,up,impl=decode_int8_impl,block_n=bn)
            return self.down_proj(y.view(x.shape[0],1,-1))
        if prefill_mlp and x.ndim == 3 and x.shape[1] > 1 and x.shape[-1] == 2048:
            m=x.shape[0]*x.shape[1]
            # Explicit A6000 candidates, not a claim of universal speedup.
            if m >= 512:
                cfg=dict(bm=64,bn=128,bk=32,warps=8,stages=3)
            else:
                cfg=dict(bm=64,bn=64,bk=64,warps=4,stages=3)
            y=gate_up_tuned(x.reshape(m,2048).contiguous(),self.gate_proj.weight,
                           self.up_proj.weight,hf_round=True,**cfg)
            return self.down_proj(y.view(*x.shape[:2],-1))
        return _mlp_decode_forward(self,x) if decode_mlp else _mlp_forward(self,x)
    return forward


def quantize_decode_mlp(model, group_size=64):
    """Optional approximate decode-only gate/up snapshot. Originals retained."""
    return {m:(quantize_groupwise(m.gate_proj.weight,group_size),
               quantize_groupwise(m.up_proj.weight,group_size))
            for m in model.modules() if isinstance(m,hf.Qwen3MLP)}


def make_lm_head_forward(impl):
    def forward(self,x):
        if impl=='contiguous':
            # Last-token slicing of [B,S,H] retains stride S*H across batch.
            # A tiny copy may let native Linear use one matrix multiply rather
            # than a strided/batched path. Keep native math and original weights.
            return torch.nn.functional.linear(x.contiguous(),self.weight,self.bias)
        if x.ndim in (2,3) and (x.ndim==2 or x.shape[1]==1) and 1<=x.shape[0]<=8:
            weight=self._int8_lm_head if impl=='int8' else self.weight
            # Weight reuse across batch rows was measured separately; without
            # it W8 LM head regressed badly for B=2/4.
            bn,warps=((8 if x.shape[0]>=4 else 4),4) if impl=='int8' else (1,4 if x.shape[0]>=4 else 8)
            out=linear_decode(x.reshape(-1,x.shape[-1]).contiguous(),weight,
                              block_n=bn,num_warps=warps,share_batch=True)
            return out.view(*x.shape[:-1],-1)
        return torch.nn.functional.linear(x,self.weight,self.bias)
    return forward


def pack_qkv_weights(model):
    """Offline inference snapshot; rebuild after changing/loading weights.

    Keeps original parameters intact, so packing costs extra resident memory.
    Passing this mapping back into the context avoids packing every iteration.
    """
    packed={}
    for module in model.modules():
        if isinstance(module,hf.Qwen3Attention):
            projections=(module.q_proj,module.k_proj,module.v_proj)
            weight=torch.cat([p.weight.detach() for p in projections],dim=0)
            if any(p.bias is not None for p in projections):
                if not all(p.bias is not None for p in projections):
                    raise ValueError('Mixed Q/K/V bias is unsupported')
                bias=torch.cat([p.bias.detach() for p in projections],dim=0)
            else:
                bias=None
            packed[module]=(weight,bias)
    return packed


def _rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if unsqueeze_dim != 1:
        raise ValueError("HF adapter supports BHSD attention only")
    result = []
    for x in (q, k):
        b, h, t, d = x.shape
        c = cos.expand(b, t, d).contiguous()
        s = sin.expand(b, t, d).contiguous()
        y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        _rotary[(triton.cdiv(x.numel(), 512),)](
            x, c, s, y, h, t, d, *x.stride(), x.numel(), 512,
            enable_fp_fusion=False)
        result.append(y)
    return tuple(result)


def _layer_addnorm_forward(self, hidden_states, attention_mask=None,
                          position_ids=None, past_key_value=None, output_attentions=False,
                          use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    attention, weights = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask,
        position_ids=position_ids, past_key_value=past_key_value,
        output_attentions=output_attentions, use_cache=use_cache,
        cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
    residual, normalized = add_rms_norm_triton(
        attention, residual, self.post_attention_layernorm.weight,
        self.post_attention_layernorm.variance_epsilon)
    out = residual + self.mlp(normalized)
    return (out, weights) if output_attentions else (out,)


@contextmanager
def qwen3_triton_inference(model, *, fuse_residual=False, decode_mlp=False, multirow_norm=False,
                          fuse_qk=False, compact_gqa=False,packed_qkv=False,packed_weights=None,
                          prefill_mlp=False,static_decode=False,direct_k_cache=False,
                          direct_v_cache=False,decode_int8_group=0,quantized_mlp=None,
                          decode_int8_impl='naive',decode_int8_max_batch=2,
                          lm_head_impl='native',lm_head_group=64,quantized_lm_head=None):
    """Scoped inference ablations; all new cache/packing/GEMM options opt-in.

    Default keeps HF cache/Linear. Static decode uses the separate runner and
    a pinned native Flash API, not HF generate() or a production serving path.
    """
    if transformers.__version__ != "4.51.0" or model.config.model_type != "qwen3":
        raise ValueError("Validated contract: Qwen3 / Transformers 4.51.0")
    if model.training or torch.is_grad_enabled():
        raise ValueError("Use model.eval() and torch.inference_mode()")
    p = next(model.parameters())
    if not p.is_cuda or p.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CUDA FP16/BF16 required")
    if (fuse_qk or compact_gqa or static_decode) and (model.config._attn_implementation!='sdpa'
                                   or getattr(model.config,'use_sliding_window',False)):
        raise ValueError('Attention optimization requires SDPA without sliding window')
    if packed_qkv and not fuse_qk:
        raise ValueError('Packed QKV requires stride-aware fused QK epilogue')
    if static_decode and not (fuse_qk and compact_gqa):
        raise ValueError('Static decode requires fuse_qk and compact_gqa')
    if static_decode and model.config.rope_scaling is not None:
        raise ValueError('Static decode validated only for default RoPE')
    if direct_k_cache and not static_decode:
        raise ValueError('Direct K store requires static_decode')
    if direct_v_cache and not direct_k_cache:
        raise ValueError('Direct V store requires direct_k_cache')
    if decode_int8_group and (not decode_mlp or decode_int8_group not in (16,32,64,128)):
        raise ValueError('INT8 decode requires decode_mlp and group=16/32/64/128')
    if decode_int8_impl not in ('naive','grouped'):
        raise ValueError('INT8 implementation must be naive or grouped')
    if decode_int8_max_batch not in (1,2):
        raise ValueError('INT8 decode batch limit must be 1 or 2')
    if lm_head_impl not in ('native','contiguous','triton','int8'):
        raise ValueError('LM head must be native, contiguous, triton, or int8')
    if lm_head_impl!='native' and model.lm_head.bias is not None:
        raise ValueError('Custom LM head requires bias-free projection')
    head_snapshot=None
    if lm_head_impl=='int8':
        head_snapshot=quantize_groupwise(model.lm_head.weight,lm_head_group) if quantized_lm_head is None else quantized_lm_head
        if head_snapshot.group_size!=lm_head_group or head_snapshot.values.shape!=model.lm_head.weight.shape:
            raise ValueError('LM head snapshot shape/group mismatch')
    quantized=quantize_decode_mlp(model,decode_int8_group) if decode_int8_group and quantized_mlp is None else quantized_mlp
    if decode_int8_group:
        for module in model.modules():
            if isinstance(module,hf.Qwen3MLP):
                if module not in quantized or any(q.group_size!=decode_int8_group for q in quantized[module]):
                    raise ValueError('Quantized MLP mapping/group mismatch')
    packed=pack_qkv_weights(model) if packed_qkv and packed_weights is None else packed_weights
    if packed_qkv and any(module not in packed for module in model.modules()
                         if isinstance(module,hf.Qwen3Attention)):
        raise ValueError('Packed weights must cover every attention layer of this model')
    attention_fn=make_attention_forward(fuse_qk=fuse_qk,compact_gqa=compact_gqa,
                                        packed_qkv=packed_qkv,direct_k_cache=direct_k_cache,
                                        direct_v_cache=direct_v_cache)
    saved = []
    saved_packed=[]
    saved_quantized=[]
    old_head=model.lm_head.__dict__.get('_int8_lm_head')
    original_mask=model.model.__dict__.get('_update_causal_mask')
    original_rope = hf.apply_rotary_pos_emb
    try:
        if static_decode:
            model.model._update_causal_mask=MethodType(
                static_mask_or_original(model.model._update_causal_mask),model.model)
        for module in model.modules():
            fn = _norm_forward if isinstance(module, hf.Qwen3RMSNorm) else (
                _mlp_forward if isinstance(module, hf.Qwen3MLP) else None)
            if decode_mlp and isinstance(module, hf.Qwen3MLP):
                fn = _mlp_decode_forward
            if (prefill_mlp or decode_int8_group) and isinstance(module,hf.Qwen3MLP):
                fn=make_mlp_forward(decode_mlp=decode_mlp,prefill_mlp=prefill_mlp,
                                    decode_int8=bool(decode_int8_group),decode_int8_impl=decode_int8_impl,
                                    decode_int8_max_batch=decode_int8_max_batch)
                if decode_int8_group:
                    saved_quantized.append((module,module.__dict__.get('_int8_gate_up')))
                    module._int8_gate_up=quantized[module]
            if multirow_norm and isinstance(module,hf.Qwen3RMSNorm):
                fn = _norm_multirow_forward
            if fuse_residual and isinstance(module, hf.Qwen3DecoderLayer):
                fn = _layer_addnorm_forward
            if (fuse_qk or compact_gqa) and isinstance(module,hf.Qwen3Attention):
                fn=attention_fn
                if packed_qkv:
                    saved_packed.append((module,module.__dict__.get('_packed_qkv')))
                    module._packed_qkv=packed[module]
            if module is model.lm_head and lm_head_impl!='native':
                fn=make_lm_head_forward(lm_head_impl)
                if head_snapshot is not None:module._int8_lm_head=head_snapshot
            if fn is not None:
                saved.append((module, module.__dict__.get("forward")))
                module.forward = MethodType(fn, module)
        hf.apply_rotary_pos_emb = _rope
        yield
    finally:
        hf.apply_rotary_pos_emb = original_rope
        if static_decode:
            if original_mask is None:
                del model.model._update_causal_mask
            else:
                model.model._update_causal_mask=original_mask
        for module,old in reversed(saved_packed):
            if old is None:
                del module._packed_qkv
            else:
                module._packed_qkv=old
        for module,old in reversed(saved_quantized):
            if old is None:
                del module._int8_gate_up
            else:
                module._int8_gate_up=old
        if head_snapshot is not None:
            if old_head is None:del model.lm_head._int8_lm_head
            else:model.lm_head._int8_lm_head=old_head
        for module, old in reversed(saved):
            if old is None:
                del module.forward
            else:
                module.forward = old
