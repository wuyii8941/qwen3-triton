"""Opt-in uninitialized storage for outputs proven completely overwritten.

Global deterministic algorithms and fill_uninitialized_memory stay enabled.
Only audited callers use these helpers. torch.empty(0).set_(storage) creates a
contiguous tensor without initializing its data; the next producer must write
EVERY element before any read. This must never allocate accumulation inputs,
partially written tensors, parameters or optimizer state.
"""
import math
import torch

_enabled = False
_calls = 0
_bytes = 0


def configure(enabled):
    global _enabled
    _enabled = bool(enabled)


def raw_overwrite_output(shape, *, device, dtype):
    shape=tuple(shape)
    if any(type(n) is not int or n<0 for n in shape):
        raise ValueError('Concrete nonnegative dimensions required')
    if torch.device(device).type!='cuda' or dtype not in (torch.float32,torch.bfloat16,torch.float16):
        raise ValueError('Audited CUDA floating point outputs only')
    size=math.prod(shape)*dtype.itemsize
    storage=torch.UntypedStorage(size,device=device)
    return torch.empty(0,device=device,dtype=dtype).set_(storage,0,shape)


def empty(shape, *, device, dtype):
    global _calls,_bytes
    if not _enabled:return torch.empty(shape,device=device,dtype=dtype)
    output=raw_overwrite_output(shape,device=device,dtype=dtype)
    _calls+=1;_bytes+=output.numel()*output.element_size()
    return output


def empty_like(tensor):
    if not tensor.is_contiguous():raise ValueError('Audited contiguous output layout required')
    return empty(tuple(tensor.shape),device=tensor.device,dtype=tensor.dtype)


def report():
    return dict(enabled=_enabled,calls=_calls,requested_bytes=_bytes,
                deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
                deterministic_fill_enabled=torch.utils.deterministic.fill_uninitialized_memory,
                scope='GEMM result and precision gate/up/SwiGLU outputs only; all coordinates written before read. Deterministic fill stays globally enabled.')
