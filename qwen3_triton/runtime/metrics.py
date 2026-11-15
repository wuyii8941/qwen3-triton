"""GPU FP64 validation statistics, preserving the CPU reference arithmetic."""
import triton
import triton.language as tl
import torch


@triton.jit
def _runtime_validation_accumulate(LOSS,TOTAL,TOKENS):
    value=tl.load(LOSS).to(tl.float64);tokens=TOKENS.to(tl.float64)
    nll=tl.load(TOTAL);count=tl.load(TOTAL+1)
    tl.store(TOTAL,nll+value*tokens);tl.store(TOTAL+1,count+tokens)


def accumulate_validation(totals,loss,tokens):
    if totals.numel()!=2 or not totals.is_cuda or totals.dtype!=torch.float64 or totals.device!=loss.device or loss.numel()!=1:
        raise ValueError('Two GPU FP64 totals and a local scalar loss required')
    if tokens<0:raise ValueError('Nonnegative token count required')
    if tokens:_runtime_validation_accumulate[(1,)](loss,totals,tokens,enable_fp_fusion=False)
