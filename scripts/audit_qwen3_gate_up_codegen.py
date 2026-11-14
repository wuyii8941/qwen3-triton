"""CPU-only comparison of fused MLP dot accumulation forms on SM86."""
from pathlib import Path
import argparse, hashlib, json, re
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from qwen3_triton.kernels.mlp_precision import _gate_up
from qwen3_triton.kernels.gemm_experiments import _gate_up_accum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    assert not out.is_relative_to(Path('/home/tzh'))
    out.mkdir(parents=True,exist_ok=False)
    assert not torch.cuda.is_initialized()
    signature={p:'*bf16' for p in ('X','WG','WU','G','U','Y')}
    constants=dict(M=512,N=12288,K=4096,BM=64,BN=64,BK=32,GROUP=8)
    attrs={(i,):[('tt.divisibility',16)] for i in range(6)}
    rows=[]
    for name,fn in (('v9',_gate_up),('explicit_accum',_gate_up_accum)):
        compiled=triton.compile(ASTSource(fn,signature,constexprs=constants,attrs=attrs),
            target=GPUTarget('cuda',86,32),options=dict(num_warps=4,num_stages=3,enable_fp_fusion=False))
        for suffix in ('ptx','ttgir'):(out/(name+'.'+suffix)).write_text(compiled.asm[suffix])
        ptx=compiled.asm['ptx'];ir=compiled.asm['ttgir']
        rows.append(dict(name=name,shared_bytes=compiled.metadata.shared,
            mma_count=ptx.count('mma.sync.'),
            total_scalar_fp32_adds=len(re.findall(r'\badd(?:\.rn)?\.f32\b',ptx)),
            float_add_IR=[line.strip() for line in ir.splitlines() if 'arith.addf' in line],
            dot_IR=[line.strip() for line in ir.splitlines() if 'tt.dot' in line],
            ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest()))
    assert not torch.cuda.is_initialized()
    report=dict(complete=True,GPU_executed=False,CUDA_initialized=False,
        performance_proven=False,numerical_equivalence_proven=False,rows=rows,
        shape=constants,scope=__doc__)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
