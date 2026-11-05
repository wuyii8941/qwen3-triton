"""Select W8 candidates before another expensive full-model integration."""
import json
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.runtime.errors import OutOfResources
from ops.triton.weight_only_gemv import quantize_groupwise,gate_up_int8
from ops.triton.weight_only_tuning import gate_up_int8_tuned
from ops.triton.decode_gate_up import decode_gate_up_triton
from benchmarks.benchmark_kernel_tuning import run_case


@torch.inference_mode()
def main():
    torch.manual_seed(744)
    g=torch.randn(6144,2048,device='cuda',dtype=torch.float16)*.02
    u=torch.randn_like(g)*.02
    qg,qu=quantize_groupwise(g,64),quantize_groupwise(u,64)
    rows=[]
    rejected=[]
    for batch in (1,2):
        x=torch.randn(batch,2048,device='cuda',dtype=torch.float16)
        expected=F.silu(F.linear(x,qg.dequantize()))*F.linear(x,qu.dequantize())
        variants={}
        for bn in (1,2,4,8):
            for warps in (4,8):
                variants[f'naive_n{bn}_w{warps}']=lambda bn=bn,warps=warps:gate_up_int8(x,qg,qu,block_n=bn,num_warps=warps)
                variants[f'grouped_n{bn}_w{warps}']=lambda bn=bn,warps=warps:gate_up_int8_tuned(x,qg,qu,bn=bn,warps=warps)
        for bn,bk,warps,stages in [(32,32,4,2),(32,64,4,3),(64,64,4,3),(128,64,4,3),(64,128,4,3),(32,128,4,3),(64,64,8,3)]:
            variants[f'tc_n{bn}_k{bk}_w{warps}_s{stages}']=lambda bn=bn,bk=bk,warps=warps,stages=stages:gate_up_int8_tuned(x,qg,qu,impl='tensorcore',bn=bn,bk=bk,warps=warps,stages=stages)
        feasible={}
        for name,fn in variants.items():
            try:
                fn()
                feasible[name]=fn
            except OutOfResources as exc:
                rejected.append(dict(batch=batch,backend=name,reason=str(exc)))
                print('unsupported',name,str(exc),flush=True)
        rows+=run_case(feasible,expected,dict(batch=batch,dtype='float16',n=6144,k=2048,group=64,scope='quantized_weight_oracle'))
        # Separate FP16 control: don't confuse quantization error with kernel error.
        control={'fp16':lambda:decode_gate_up_triton(x,g,u,block_n=1)}
        rows+=run_case(control,F.silu(F.linear(x,g))*F.linear(x,u),dict(batch=batch,dtype='float16',scope='original_fp16_control'))
    target=Path('artifacts/weight_only_tuning.json')
    if target.exists(): raise FileExistsError(target)
    sources=['ops/triton/weight_only_gemv.py','ops/triton/weight_only_tuning.py',__file__]
    target.write_text(json.dumps(dict(seed=744,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        rows=rows,rejected=rejected,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
    for b in (1,2):
        print('batch',b,[(r['backend'],round(r['median_us'],2)) for r in sorted((r for r in rows if r['batch']==b),key=lambda r:r['median_us'])[:8]],flush=True)


if __name__=='__main__': main()
