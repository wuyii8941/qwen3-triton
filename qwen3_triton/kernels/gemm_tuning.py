"""Explicit opt-in configuration plan. Disabled unless the caller loads a plan."""
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager

_entries = {}
_identity = None
_enabled = True
_capabilities = {}


@contextmanager
def enabled(value):
    """Explicit control for same-process V9/candidate module comparisons."""
    global _enabled
    before = _enabled
    _enabled = bool(value)
    try:
        yield
    finally:
        _enabled = before


def load_plan(path):
    global _entries, _identity
    if _identity is not None:
        raise RuntimeError('Load a GEMM plan only once, before constructing the model')
    path = Path(path)
    raw = path.read_bytes()
    plan = json.loads(raw)
    if (plan['schema'] != 1 or plan['hardware_capability'] != [8,6]
            or plan['compute_dtype'] != 'bfloat16' or plan['accumulation_dtype'] != 'float32'):
        raise ValueError('Unsupported GEMM plan')
    entries = {}
    for entry in plan['entries']:
        cfg = tuple(entry['config'])
        if len(cfg) != 6 or any(type(v) is not int or v <= 0 for v in cfg):
            raise ValueError('Invalid configuration')
        tm, tn, tk, group, warps, stages = cfg
        if (tm not in (16,32,64,128) or tn not in (32,64,128,256)
                or tk not in (32,64,128) or group not in (1,4,8)
                or warps not in (4,8) or stages not in (2,3,4)):
            raise ValueError('Configuration outside screened search space')
        if entry['family'] not in ('tiled','split_k') or entry['splits'] not in (1,2,4,8):
            raise ValueError('Unsupported algorithm')
        if (entry['family'] == 'split_k') != (entry['splits'] > 1):
            raise ValueError('Inconsistent reduction decomposition')
        if entry['kind'] not in ('gemm','gate_up'):
            raise ValueError('Unknown operator')
        if entry['kind'] == 'gate_up' and entry['family'] != 'tiled':
            raise ValueError('Unsupported gate/up decomposition')
        strides = () if entry['strides'] is None else tuple(v for t in entry['strides'] for v in t)
        key = (entry['kind'],entry['M'],entry['N'],entry['K'],*strides)
        if key in entries:
            raise ValueError('Duplicate shape/layout')
        entries[key] = entry
    _entries = entries
    _identity = dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),entries=len(entries))
    return dict(_identity)


def identity():
    return None if _identity is None else dict(_identity)


def gemm_entry(a,b,output_dtype):
    if not _enabled or not _entries:
        return None
    import torch
    if a.dtype != torch.bfloat16 or output_dtype != torch.bfloat16:
        return None
    m,k = a.shape
    key = ('gemm',m,b.shape[1],k,*a.stride(),*b.stride())
    entry = _entries.get(key)
    if entry is None:
        return None
    device = a.device.index
    capability = _capabilities.get(device)
    if capability is None:
        capability = torch.cuda.get_device_capability(a.device)
        _capabilities[device] = capability
    return entry if capability == (8,6) else None


def gate_up_entry(x,n):
    if not _enabled or not _entries:
        return None
    import torch
    if x.dtype != torch.bfloat16:
        return None
    m = x.numel()//x.shape[-1]
    entry = _entries.get(('gate_up',m,n,x.shape[-1]))
    if entry is None:
        return None
    device = x.device.index
    capability = _capabilities.get(device)
    if capability is None:
        capability = torch.cuda.get_device_capability(x.device)
        _capabilities[device] = capability
    return entry if capability == (8,6) else None
