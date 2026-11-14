"""CPU-only SM86 resource/code-generation screen. No GPU timing or quality claim."""
from pathlib import Path
import argparse, json, re
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from qwen3_triton.kernels.linear import _gemm
from qwen3_triton.kernels.mlp_precision import _gate_up
from tune_qwen3_ampere_gemm import CONFIGS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.resolve().is_relative_to(Path('/home/tzh'))
    args.output.mkdir(parents=True,exist_ok=False)
    assert not torch.cuda.is_initialized()
    rows=[]
    cases=[('head_forward',512,151936,4096,4096,1,1,4096),
           ('head_dx',512,4096,151936,151936,1,4096,1),
           ('head_dw',151936,4096,512,1,151936,4096,1),
           ('mlp_dx',512,4096,12288,12288,1,4096,1),
           ('mlp_dw',12288,4096,512,1,12288,4096,1),
           ('gate_up',512,12288,4096,4096,1,1,4096)]
    for name,m,n,k,am,ak,bk,bn in cases:
        for cfg in CONFIGS:
            tm,tn,tk,group,warps,stages=cfg
            fused=name=='gate_up'
            if (tm+(2*tn if fused else tn))*tk*2*stages>99*1024:
                continue
            row=dict(case=name,config=cfg)
            signature={p:'*bf16' for p in (('X','WG','WU','G','U','Y') if fused else ('A','B','C'))}
            constants=dict(M=m,N=n,K=k)
            if fused:
                constants.update(BM=tm,BN=tn,BK=tk,GROUP=group)
            else:
                constants.update(AM=am,AK=ak,BK=bk,BN=bn,BM=tm,BN_TILE=tn,BK_TILE=tk,GROUP=group)
            attrs={(i,):[('tt.divisibility',16)] for i in range(len(signature))}
            try:
                compiled=triton.compile(ASTSource(_gate_up if fused else _gemm,signature,constexprs=constants,attrs=attrs),
                    target=GPUTarget('cuda',86,32),options=dict(num_warps=warps,num_stages=stages,enable_fp_fusion=not fused))
                ptx=compiled.asm['ptx'];ir=compiled.asm['ttgir']
                row.update(compiled=True,shared_bytes=compiled.metadata.shared,
                    tensorcore_instructions=ptx.count('mma.sync.'),
                    async_copy_instructions=ptx.count('cp.async.'),
                    scalar_fp32_adds=len(re.findall(r'\badd(?:\.rn)?\.f32\b',ptx)),
                    local_memory_in_ptx=bool(re.search(r'\b(?:ld|st)\.local\b',ptx)))
            except triton.OutOfResources as exc:
                row.update(compiled=False,resource_rejected=str(exc))
            rows.append(row)
            (args.output/'progress.json').write_text(json.dumps(dict(complete=False,rows=rows),indent=2)+'\n')
        print(json.dumps(dict(case=name,configs=sum(r['case']==name for r in rows))),flush=True)
    assert not torch.cuda.is_initialized()
    (args.output/'result.json').write_text(json.dumps(dict(complete=True,GPU_executed=False,
        CUDA_initialized=False,target='SM86',rows=rows,scope=__doc__),indent=2)+'\n')


if __name__=='__main__':main()
