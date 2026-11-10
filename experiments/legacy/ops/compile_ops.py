"""Inference-only opaque operator boundaries for pinned torch 2.5.1.

Its Triton mutation analyzer misinterprets the launch option
``enable_fp_fusion=False`` as a kernel argument. Explicit schemas avoid the
conservative 'all inputs mutated' fallback without changing numeric rounding.
Eager paths still launch their kernels directly; only compiled paths use these
operators. No autograd support is claimed. Cache writes declare their mutation;
outputs are fresh tensors, never aliases of inputs.
"""
import torch
from torch import Tensor


@torch.library.custom_op('qwen_lab::norm', mutates_args=(), device_types='cuda')
def norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    from model.hf_inference import _norm_impl
    return _norm_impl(x, weight, eps)


@norm.register_fake
def _(x, weight, eps):
    return torch.empty_like(x.contiguous())


@torch.library.custom_op('qwen_lab::activation', mutates_args=(), device_types='cuda')
def activation(gate: Tensor, up: Tensor) -> Tensor:
    from model.hf_inference import _activation_impl
    return _activation_impl(gate, up)


@activation.register_fake
def _(gate, up):
    return torch.empty_like(gate)


@torch.library.custom_op('qwen_lab::add_norm', mutates_args=(), device_types='cuda')
def add_norm(x: Tensor, residual: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    from ops.triton.add_rms_norm import add_rms_norm_triton
    return add_rms_norm_triton(x, residual, weight, eps)


@add_norm.register_fake
def _(x, residual, weight, eps):
    return torch.empty_like(x.contiguous()), torch.empty_like(x.contiguous())


@torch.library.custom_op('qwen_lab::decode_gate_up', mutates_args=(), device_types='cuda')
def decode_gate_up(x: Tensor, gate: Tensor, up: Tensor, bn: int, warps: int) -> Tensor:
    from ops.triton.decode_gate_up import decode_gate_up_triton
    return decode_gate_up_triton(x, gate, up, block_n=bn, num_warps=warps)


@decode_gate_up.register_fake
def _(x, gate, up, bn, warps):
    return x.new_empty((x.shape[0], gate.shape[0]))


@torch.library.custom_op('qwen_lab::gate_up', mutates_args=(), device_types='cuda')
def gate_up(x: Tensor, gate: Tensor, up: Tensor, bm: int, bn: int, bk: int,
            warps: int, stages: int, group_m: int, hf_round: bool) -> Tensor:
    from ops.triton.gemm_tuning import gate_up_tuned
    return gate_up_tuned(x, gate, up, bm=bm, bn=bn, bk=bk, warps=warps,
                         stages=stages, group_m=group_m, hf_round=hf_round)


@gate_up.register_fake
def _(x, gate, up, bm, bn, bk, warps, stages, group_m, hf_round):
    return x.new_empty((x.shape[0], gate.shape[0]))


@torch.library.custom_op('qwen_lab::qk_rope', mutates_args=(), device_types='cuda')
def qk_rope(q: Tensor, k: Tensor, wq: Tensor, wk: Tensor, cos: Tensor,
            sin: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    from ops.triton.hf_qk_norm_rope import _launch
    return _launch(q, k, wq, wk, cos, sin, eps)


@qk_rope.register_fake
def _(q, k, wq, wk, cos, sin, eps):
    return (q.new_empty((q.shape[0], q.shape[2], q.shape[1], q.shape[3])),
            k.new_empty((k.shape[0], k.shape[2], k.shape[1], k.shape[3])))


@torch.library.custom_op('qwen_lab::qk_rope_store_k', mutates_args=('key_cache',), device_types='cuda')
def qk_rope_store_k(q: Tensor, k: Tensor, wq: Tensor, wk: Tensor, cos: Tensor,
                    sin: Tensor, eps: float, key_cache: Tensor, positions: Tensor) -> Tensor:
    from ops.triton.hf_qk_norm_rope import _launch
    return _launch(q, k, wq, wk, cos, sin, eps, k_cache=key_cache, positions=positions)[0]


@qk_rope_store_k.register_fake
def _(q, k, wq, wk, cos, sin, eps, key_cache, positions):
    return q.new_empty((q.shape[0], q.shape[2], q.shape[1], q.shape[3]))


@torch.library.custom_op('qwen_lab::qk_rope_store_kv',
                          mutates_args=('key_cache', 'value_cache'), device_types='cuda')
def qk_rope_store_kv(q: Tensor, k: Tensor, wq: Tensor, wk: Tensor, cos: Tensor,
                     sin: Tensor, eps: float, key_cache: Tensor, positions: Tensor,
                     v: Tensor, value_cache: Tensor) -> Tensor:
    from ops.triton.hf_qk_norm_rope import _launch
    return _launch(q, k, wq, wk, cos, sin, eps, k_cache=key_cache,
                   positions=positions, v=v, v_cache=value_cache)[0]


@qk_rope_store_kv.register_fake
def _(q, k, wq, wk, cos, sin, eps, key_cache, positions, v, value_cache):
    return q.new_empty((q.shape[0], q.shape[2], q.shape[1], q.shape[3]))
