"""Read-only exact-BF16 provenance check with one device flag as workspace.

Independent experiment, not yet integrated into packing or FSDP. The returned
flag may be MIN-reduced collectively before the caller decides to reject.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _validate_upcast(x, flag, count, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = i < count
    value = tl.load(x + i, mask=active, other=0)
    bits = value.to(tl.uint32, bitcast=True)
    # BF16->FP32 preserves all exponent bits and fills the low16 with zeros.
    # Reject both NaN and infinity; signed zero/subnormals remain valid.
    invalid = active & (((bits & 65535) != 0) | ((bits & 0x7f800000) == 0x7f800000))
    if tl.sum(invalid.to(tl.int32), axis=0) != 0:
        tl.atomic_xchg(flag, 0)


def validate_upcast_flag(x):
    """Return a CUDA int32 scalar:1 if all entries are finite BF16 upcasts.

    No host synchronization or large temporary tensors. Work is enqueued on
    the current CUDA stream; consumers must honor stream/collective ordering.
    This validates VALUES only, not ownership, lifetime or aliasing.
    """
    if (x.dtype != torch.float32 or not x.is_cuda or x.ndim != 1
            or not x.is_contiguous() or not x.numel()):
        raise ValueError('Nonempty contiguous one-dimensional FP32 CUDA input required')
    flag = torch.ones((), dtype=torch.int32, device=x.device)
    _validate_upcast[(triton.cdiv(x.numel(), 4096),)](x, flag, x.numel(), BLOCK=4096, num_warps=4)
    return flag


def reference_upcast_valid(x):
    """Small CPU oracle; deliberately not the low-workspace GPU path."""
    if x.device.type != 'cpu' or x.dtype != torch.float32:
        raise ValueError('FP32 CPU oracle input required')
    return bool(torch.isfinite(x).all() and ((x.contiguous().view(torch.int32) & 65535) == 0).all())
