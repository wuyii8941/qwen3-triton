"""Native low-precision oracle with the SAME microbatch partition as FSDP.

Different GEMM shapes need not be bitwise equivalent in BF16. Full-batch
mathematical normalization equivalence is a separate FP32/CPU DDP check.
"""
from types import SimpleNamespace
import torch


def native_microbatch_reference(model, ids, *, dtype, tied, accumulation=1):
    if type(accumulation) is not int or accumulation < 1 or ids.shape[0] % accumulation:
        raise ValueError('Positive accumulation must divide the reference batch')
    parts = []
    for micro in ids.chunk(accumulation):
        # Fresh casts per microbatch: otherwise BF16 shadow gradients could
        # accumulate before the FP32 master cast, unlike per-forward FSDP.
        casts = {n: p.to(dtype) for n, p in model.named_parameters(remove_duplicate=tied)}
        parts.append(torch.func.functional_call(model, casts, (micro,),
            dict(labels=micro, use_cache=False), tie_weights=tied))
    if accumulation == 1:
        return parts[0]
    return SimpleNamespace(logits=torch.cat([p.logits for p in parts]),
                           loss=torch.stack([p.loss for p in parts]).mean())
