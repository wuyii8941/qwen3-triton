"""Destructive, experimental packing of explicitly owned FP32 scratch.

NOT connected to FSDP. No claim of CUDA correctness or speed until GPU gates.
The input must be an exact finite BF16 upcast. Packing destroys FP32 values.
CPU is a scheduling/bit-level oracle; CUDA uses ordered launches on one stream.
"""
import sys

import torch
import triton
import triton.language as tl


def packing_stages(count):
    """Return [lo, hi) stages with globally disjoint read/write addresses.

    BF16 source element i lives at word 2*i+1; destination lives at word i.
    hi <= 2*lo+1 ensures max destination < min source within each launch.
    Earlier stages wrote only indices < lo, also disjoint from later sources.
    """
    if type(count) is not int or count < 1:
        raise ValueError('Positive integer count required')
    lo = 0
    while lo < count:
        hi = min(count, 2 * lo + 1)
        yield lo, hi
        lo = hi


@triton.jit
def _compact_stage(words, lo, hi, BLOCK: tl.constexpr):
    i = lo + tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(words + 2 * i + 1, mask=i < hi, other=0)
    tl.store(words + i, value, mask=i < hi)


@torch.no_grad()
def pack_owned_scratch(scratch, *, allow_mutation=False, backend='cpu'):
    """Return disjoint BF16 send/receive views in the same 4*N-byte storage.

    Explicit ownership is a CALLER obligation, not something tensor metadata
    can prove. No aliases may be used concurrently. Validation intentionally
    synchronizes CUDA; this is a correctness prototype, not a timed callback.
    Receive contents are unspecified and must be overwritten by transport.
    """
    if allow_mutation is not True:
        raise ValueError('Explicit ownership and allow_mutation=True required')
    if (sys.byteorder != 'little' or scratch.dtype != torch.float32
            or scratch.ndim != 1 or not scratch.is_contiguous()
            or scratch.storage_offset() != 0 or not scratch.numel()
            or scratch.requires_grad or scratch._base is not None
            or scratch.untyped_storage().nbytes() != scratch.numel() * 4):
        raise ValueError('Standalone contiguous finite BF16-upcast FP32 scratch required')
    if backend not in ('cpu', 'triton'):
        raise ValueError('Unknown backend')
    if (backend == 'cpu' and scratch.device.type != 'cpu') or (
            backend == 'triton' and not scratch.is_cuda):
        raise ValueError('Backend/device mismatch')
    bits = scratch.view(torch.int32)
    if not torch.isfinite(scratch).all().item() or ((bits & 65535) != 0).any().item():
        raise ValueError('Only exact finite BF16 upcasts may be packed')
    words = scratch.view(torch.bfloat16)
    count = scratch.numel()
    for lo, hi in packing_stages(count):
        if backend == 'cpu':
            words[lo:hi].copy_(words[2 * lo + 1:2 * hi:2])
        else:
            _compact_stage[(triton.cdiv(hi - lo, 1024),)](words, lo, hi, BLOCK=1024)
    return words[:count], words[count:]
