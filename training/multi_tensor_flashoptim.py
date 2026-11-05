"""Opt-in launch batching for small BF16 FlashAdamW tensors.

Calls the installed FlashOptim JIT body unchanged. Axis 1 chooses a parameter;
axis 0 keeps its original local block indexing. No flatten/cat, no new
quantization boundaries, no precision/master-weight changes. Large or
unsupported tensors retain the official step_param path. Not a default.
"""
import torch
import triton
import triton.language as tl
from flashoptim import FlashAdamW
from flashoptim.optimizers import _triton_adam_kernel


@triton.jit
def _batched_update(TABLE, LR, B1, B2, EPS, WD, BLOCK: tl.constexpr):
    row = tl.program_id(1) * 8
    p = tl.load(TABLE + row).to(tl.pointer_type(tl.bfloat16))
    g = tl.load(TABLE + row + 1).to(tl.pointer_type(tl.bfloat16))
    m = tl.load(TABLE + row + 2).to(tl.pointer_type(tl.int8))
    ms = tl.load(TABLE + row + 3).to(tl.pointer_type(tl.float16))
    v = tl.load(TABLE + row + 4).to(tl.pointer_type(tl.uint8))
    vs = tl.load(TABLE + row + 5).to(tl.pointer_type(tl.float16))
    n = tl.load(TABLE + row + 6)
    step = tl.load(TABLE + row + 7).to(tl.int32)
    _triton_adam_kernel(m, ms, v, vs, p, g, p, n, LR, B1, B2, EPS, WD, step,
                       GROUP_SIZE=32, DECOUPLED_WEIGHT_DECAY=True,
                       PARAM_DTYPE=tl.bfloat16, USE_ECC=False,
                       QUANTIZE_OPTIM_STATES=True, SIGNED_ERROR_T=tl.int8,
                       NUM_MANTISSA_BITS=7, SIGNED_MAX_VAL=127, BLOCK_SIZE_N=BLOCK)


@triton.jit
def _batched_update_vectorized(TABLE, LR, B1, B2, EPS, WD, BLOCK: tl.constexpr, ALIGNED_N: tl.constexpr):
    row=tl.program_id(1)*8
    # Pointer tables erase the host launcher's alignment specialization.
    # Restore only facts checked by the host, never guess alignment.
    p=tl.multiple_of(tl.load(TABLE+row).to(tl.pointer_type(tl.bfloat16)),16)
    g=tl.multiple_of(tl.load(TABLE+row+1).to(tl.pointer_type(tl.bfloat16)),16)
    m=tl.multiple_of(tl.load(TABLE+row+2).to(tl.pointer_type(tl.int8)),16)
    ms=tl.multiple_of(tl.load(TABLE+row+3).to(tl.pointer_type(tl.float16)),16)
    v=tl.multiple_of(tl.load(TABLE+row+4).to(tl.pointer_type(tl.uint8)),16)
    vs=tl.multiple_of(tl.load(TABLE+row+5).to(tl.pointer_type(tl.float16)),16)
    n=tl.load(TABLE+row+6).to(tl.int32)
    if ALIGNED_N: n=tl.multiple_of(n,32)
    step=tl.load(TABLE+row+7).to(tl.int32)
    _triton_adam_kernel(m,ms,v,vs,p,g,p,n,LR,B1,B2,EPS,WD,step,
        GROUP_SIZE=32,DECOUPLED_WEIGHT_DECAY=True,PARAM_DTYPE=tl.bfloat16,USE_ECC=False,
        QUANTIZE_OPTIM_STATES=True,SIGNED_ERROR_T=tl.int8,NUM_MANTISSA_BITS=7,
        SIGNED_MAX_VAL=127,BLOCK_SIZE_N=BLOCK)


class MultiTensorFlashAdamW(FlashAdamW):
    """Experimental small-tensor batching with official state_dict compatibility."""
    def __init__(self, params, *, max_numel=65536, persistent=False, vectorized=False, async_metadata=False, **kwargs):
        super().__init__(params, **kwargs)
        self.max_numel = max_numel
        self.persistent = persistent
        self.vectorized = vectorized
        self.async_metadata = async_metadata
        self._grid_caps = {}
        self._metadata_buffers = {}

    @torch.no_grad()
    def step(self, closure=None):
        if self._gradient_release or closure is not None:
            return super().step(closure)
        for group_id, group in enumerate(self.param_groups):
            buckets = {}
            for p in group['params']:
                if not p.requires_grad or p.grad is None:
                    continue
                eligible = (self._fused and not self._check_numerics and self._decoupled
                            and p.is_cuda and p.dtype == torch.bfloat16
                            and p.is_contiguous() and not p.grad.is_sparse
                            and p.grad.dtype == p.dtype and p.grad.is_contiguous()
                            and 0 < p.numel() <= min(self.max_numel,2**31-1))
                if eligible:
                    self._ensure_state_initialized(p, hparams=group)
                    state = self.state[p]
                    m, v = state['exp_avg'], state['exp_avg_sq']
                    eligible = (state.get('error_bits') is None
                                and m.is_quantized() and v.is_quantized()
                                and m.kernel_scales_or_self.dtype == torch.float16
                                and v.kernel_scales_or_self.dtype == torch.float16)
                    if eligible and self.vectorized:
                        eligible=all(t.data_ptr()%16==0 for t in (p,p.grad,m.kernel_tensor,
                            m.kernel_scales_or_self,v.kernel_tensor,v.kernel_scales_or_self))
                if not eligible:
                    self.step_param(p, group)
                    continue
                state['step'] += 1
                blocks = triton.next_power_of_2(triton.cdiv(p.numel(), 1024))
                if self.persistent:
                    device = str(p.device)
                    if device not in self._grid_caps:
                        self._grid_caps[device] = 2*torch.cuda.get_device_properties(p.device).multi_processor_count
                    # Same persistent axis-0 stride as the official launch.
                    # Axis 1 independently chooses a tensor; groups never cross tensors.
                    blocks = min(triton.cdiv(p.numel(),1024),self._grid_caps[device])
                key = (str(p.device), blocks, p.numel()%32==0 if self.vectorized else False)
                buckets.setdefault(key, []).append([
                    p.data_ptr(), p.grad.data_ptr(), m.kernel_tensor.data_ptr(),
                    m.kernel_scales_or_self.data_ptr(), v.kernel_tensor.data_ptr(),
                    v.kernel_scales_or_self.data_ptr(), p.numel(), int(state['step'].item())])
            for (device, blocks, aligned_n), rows in buckets.items():
                key = (group_id, device, blocks, aligned_n, len(rows))
                if key not in self._metadata_buffers:
                    self._metadata_buffers[key] = torch.empty((len(rows), 8), dtype=torch.int64, device=device)
                table = self._metadata_buffers[key]
                # Include this host construction/upload in optimizer timing.
                # Fresh pinned allocation is intentional: PyTorch's caching
                # host allocator tracks outstanding copies before reuse. Do
                # not overwrite a single pinned buffer while DMA is pending.
                host=torch.tensor(rows,dtype=torch.int64,pin_memory=self.async_metadata)
                table.copy_(host,non_blocking=self.async_metadata)
                wd = group['weight_decay']
                wd *= (group['lr'] / group['initial_lr']) if self._decouple_lr else group['lr']
                if self.vectorized:
                    _batched_update_vectorized[(blocks,len(rows))](table,group['lr'],*group['betas'],
                                                                 group['eps'],wd,1024,aligned_n)
                else:
                    _batched_update[(blocks, len(rows))](table, group['lr'], *group['betas'],
                                                       group['eps'], wd, 1024)
        return None
