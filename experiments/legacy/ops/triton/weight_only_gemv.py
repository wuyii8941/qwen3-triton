"""Opt-in W8A16 decode experiment; never overwrites checkpoint parameters.

Symmetric per-row/per-K-group int8 weights, FP32 scales, FP16/BF16 activations.
This changes model numerics. Kernel parity to dequantized weights is distinct
from the full-model quality gate against the original FP16 checkpoint.
"""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class GroupwiseInt8:
    values: torch.Tensor
    scales: torch.Tensor
    group_size: int
    dtype: torch.dtype

    @property
    def nbytes(self):
        return self.values.numel() + self.scales.numel() * self.scales.element_size()

    def dequantize(self):
        n, k = self.values.shape
        return (self.values.view(n, -1, self.group_size).float()
                * self.scales[:, :, None]).reshape(n, k).to(self.dtype)


def quantize_groupwise(weight, group_size=64):
    if torch.is_grad_enabled():
        raise ValueError('Quantization is an inference-only snapshot')
    if weight.ndim != 2 or not weight.is_cuda or weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('CUDA FP16/BF16 [N,K] weight required')
    if group_size not in (16, 32, 64, 128) or weight.shape[1] % group_size:
        raise ValueError('K must be divisible by group_size=16/32/64/128')
    n, k = weight.shape
    w = weight.detach().float().reshape(n, -1, group_size)
    scale = w.abs().amax(-1) / 127.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (w / scale[:, :, None]).round().clamp(-127, 127).to(torch.int8).reshape(n, k).contiguous()
    return GroupwiseInt8(q, scale.contiguous(), group_size, weight.dtype)


@triton.jit
def _w8_gate_up(X, G, U, SG, SU, Y, N: tl.constexpr, K: tl.constexpr,
                GROUP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    x = tl.load(X + m*K+k, k < K, 0).to(tl.float32)
    mask = (n[:, None] < N) & (k[None, :] < K)
    g = tl.load(G+n[:, None]*K+k[None, :], mask, 0).to(tl.float32)
    u = tl.load(U+n[:, None]*K+k[None, :], mask, 0).to(tl.float32)
    gs = tl.load(SG+n[:, None]*(K//GROUP)+k[None, :]//GROUP, mask, 0)
    us = tl.load(SU+n[:, None]*(K//GROUP)+k[None, :]//GROUP, mask, 0)
    # Define exact semantics against an explicit low-precision dequantization.
    g = (g*gs).to(X.dtype.element_ty).to(tl.float32)
    u = (u*us).to(X.dtype.element_ty).to(tl.float32)
    gate = tl.sum(g*x[None, :], 1).to(Y.dtype.element_ty).to(tl.float32)
    up = tl.sum(u*x[None, :], 1).to(Y.dtype.element_ty).to(tl.float32)
    silu = (gate/(1+tl.exp(-gate))).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+m*N+n, silu*up, n < N)


def gate_up_int8(x, gate, up, *, block_n=1, num_warps=4, impl='naive'):
    if torch.is_grad_enabled():
        raise ValueError('Inference only')
    if gate.group_size != up.group_size or gate.values.shape != up.values.shape:
        raise ValueError('Matching packed gate/up required')
    n, k = gate.values.shape
    if x.ndim != 2 or x.shape[1] != k or not 1 <= x.shape[0] <= 8 or k > 4096:
        raise ValueError('Supported X [M,K], M=1..8, K<=4096')
    if not x.is_cuda or x.dtype != gate.dtype or x.dtype != up.dtype or not x.is_contiguous():
        raise ValueError('Contiguous CUDA input with original weight dtype required')
    for w in (gate, up):
        if w.values.dtype != torch.int8 or w.scales.dtype != torch.float32:
            raise ValueError('Invalid weight/scales dtype')
        if any(t.device != x.device or not t.is_contiguous() for t in (w.values, w.scales)):
            raise ValueError('Packed weights must be contiguous and on input device')
        if w.scales.shape != (n, k // w.group_size):
            raise ValueError('Invalid scales shape')
    if block_n not in (1, 2, 4, 8) or num_warps not in (4, 8):
        raise ValueError('Unsupported launch configuration')
    if impl == 'grouped':
        from .weight_only_tuning import gate_up_int8_tuned
        return gate_up_int8_tuned(x,gate,up,bn=block_n,warps=num_warps)
    if impl != 'naive':
        raise ValueError('Expected naive or grouped implementation')
    out = torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype)
    _w8_gate_up[(x.shape[0], triton.cdiv(n, block_n))](
        x, gate.values, up.values, gate.scales, up.scales, out, n, k,
        gate.group_size, block_n, triton.next_power_of_2(k),
        num_warps=num_warps, enable_fp_fusion=False)
    return out
