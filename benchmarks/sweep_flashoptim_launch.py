"""Tune official quantized Adam kernel launch only; no formula changes."""
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from flashoptim.optimizers import _triton_adam_kernel, _make_grid


def main():
    root = Path('artifacts/flashoptim_launch_sweep_v1')
    root.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    report = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), sm=sm, cases=[])
    configs = [(1024, 1, 4), (1024, 2, 4), (1024, 4, 4), (2048, 1, 4),
               (2048, 2, 4), (2048, 4, 4), (4096, 2, 4), (4096, 2, 8)]
    for n in (2048, 2048*2048, 6144*2048, 151936*2048):
        initial = torch.randn(n, device='cuda', dtype=torch.bfloat16)*.02
        grad = torch.randn_like(initial)*.001
        def new_state():
            return [initial.clone(), torch.zeros(n, device='cuda', dtype=torch.int8),
                    torch.zeros(n, device='cuda', dtype=torch.uint8),
                    torch.full((triton.cdiv(n,32),), 1e-12, device='cuda', dtype=torch.float16),
                    torch.full((triton.cdiv(n,32),), 1e-12, device='cuda', dtype=torch.float16)]
        def update(state, step, config=None):
            p,m,v,ms,vs = state
            block, multiplier, warps = config or (1024, 1, 4)
            grid = (lambda meta: _make_grid(n, meta)) if config is None else (min(triton.cdiv(n,block), sm*multiplier),)
            _triton_adam_kernel[grid](m,ms,v,vs,p,grad,p,n,1e-5,.9,.999,1e-8,1e-7,step,
                GROUP_SIZE=32,DECOUPLED_WEIGHT_DECAY=True,PARAM_DTYPE=tl.bfloat16,USE_ECC=False,
                QUANTIZE_OPTIM_STATES=True,SIGNED_ERROR_T=tl.int8,NUM_MANTISSA_BITS=7,
                SIGNED_MAX_VAL=127,BLOCK_SIZE_N=block,num_warps=warps)
        reference = new_state()
        for i in range(1,5): update(reference,i)
        timings = []
        for config in [None] + configs:
            state = new_state()
            try:
                for i in range(1,5): update(state,i,config)
                equal = all(torch.equal(a,b) for a,b in zip(state,reference))
                row = dict(config=config, exact_after_4_steps=equal)
                if equal:
                    # Include host launch overhead; separately report CUDA-event time.
                    host, device = [], []
                    for i in range(5,23):
                        start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        start = time.perf_counter(); start_event.record()
                        update(state,i,config)
                        stop_event.record(); stop_event.synchronize()
                        if i >= 8:
                            host.append((time.perf_counter()-start)*1000)
                            device.append(start_event.elapsed_time(stop_event))
                    row.update(host_median_ms=statistics.median(host), device_median_ms=statistics.median(device))
                timings.append(row)
            except Exception as exc:
                timings.append(dict(config=config, exact_after_4_steps=False,error=str(exc)))
            del state
        report['cases'].append(dict(numel=n, timings=timings))
        (root/'result.json').write_text(json.dumps(report,indent=2))
        print('DONE',n,json.dumps(timings),flush=True)
        del initial,grad,reference
        torch.cuda.empty_cache()
    report['complete']=True
    (root/'result.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
