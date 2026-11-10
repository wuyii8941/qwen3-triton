"""Kernel-internal ablations, fixed candidates + alternating graph replay.

All variants include their own allocations/copies/merge in the captured graph.
CUDA Graph GPU time is not end-to-end model wall time. JIT/capture is excluded
from steady-state and reported separately. No silent fallback on failed checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import sdpa_kernel,SDPBackend
from ops.triton.decode_split_kv import attention_decode_split
from ops.triton.attention import attention_triton_decode
from ops.triton.rowwise_tuning import rms_norm_rows,rope_rows
from ops.triton.gemm_tuning import gate_up_tuned
from ops.triton.gate_up_swiglu import gate_up_swiglu_triton
from model.hf_inference import _norm,_rotary,_activation


def run_case(variants,expected,meta):
    graphs={}
    records={}
    # Record candidate numerical errors BEFORE collecting any speed result.
    for name,fn in variants.items():
        start=time.perf_counter()
        y=fn()
        torch.cuda.synchronize()
        first_ms=(time.perf_counter()-start)*1000
        yf,ef=y.float(),expected.float()
        if not torch.isfinite(yf).all():
            raise AssertionError((name,'nonfinite'))
        rel=float(torch.linalg.vector_norm(yf-ef)/torch.linalg.vector_norm(ef).clamp_min(1e-12))
        max_abs=float((yf-ef).abs().max())
        assert rel<.02,(name,meta,rel,max_abs)
        torch.testing.assert_close(yf,ef,atol=.025,rtol=.04)
        del y,yf,ef
        for _ in range(3): fn()
        torch.cuda.synchronize()
        base=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        y=fn()
        torch.cuda.synchronize()
        delta=torch.cuda.max_memory_allocated()-base
        del y
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10): out=fn()
        graphs[name]=(graph,out)
        records[name]=dict(**meta,backend=name,relative_l2=rel,max_abs=max_abs,
                           first_call_ms=first_ms,peak_delta_bytes=delta,samples_us=[])
    names=list(variants)
    for repeat in range(9):
        for name in (names if repeat%2==0 else names[::-1]):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            records[name]['samples_us'].append(start.elapsed_time(end)*1000/10)
    for row in records.values():
        row['median_us']=statistics.median(row['samples_us'])
        row['min_us']=min(row['samples_us'])
        row['max_us']=max(row['samples_us'])
    best=min(records,key=lambda n:records[n]['median_us'])
    print(meta,'best',best,round(records[best]['median_us'],2),
          'baselines',{n:round(r['median_us'],2) for n,r in records.items() if n in
            ('old','flash','original_autotune','cublas','row1','block512_w4')},flush=True)
    return list(records.values())


def decode_cases(dtype):
    rows=[]
    for b in (1,2):
        for seq in (128,512,2048,8192):
            q=torch.randn(b,16,1,128,device='cuda',dtype=dtype)
            k=torch.randn(b,8,seq,128,device='cuda',dtype=dtype)
            v=torch.randn_like(k)
            def flash():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(q,k,v,enable_gqa=True)
            expected=flash()
            variants={'old':lambda:attention_triton_decode(q,k,v),'flash':flash}
            for grouped in (False,True):
                for splits in (1,2,4,8,16,32):
                    if splits>triton.cdiv(seq,64): continue
                    variants[f'g{int(grouped)}_s{splits}']=lambda grouped=grouped,splits=splits:attention_decode_split(
                        q,k,v,splits=splits,group_heads=grouped)
            rows+=run_case(variants,expected,dict(op='decode',dtype=str(dtype),b=b,seq=seq))
    return rows


def row_cases(dtype):
    records=[]
    for n in (128,2048):
        for m in (16,2048,16384):
            x=torch.randn(m,n,device='cuda',dtype=dtype)
            w=torch.randn(n,device='cuda',dtype=dtype)
            expected=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(dtype)*w
            def old():
                y=torch.empty_like(x)
                _norm[(m,)](x,w,y,n,1e-6,triton.next_power_of_2(n),enable_fp_fusion=False)
                return y
            variants={'old':old}
            for r in (1,2,4,8,16):
                variants[f'row{r}']=lambda r=r:rms_norm_rows(x,w,rows=r)
            records+=run_case(variants,expected,dict(op='rmsnorm',dtype=str(dtype),m=m,n=n))
    for b,s in ((1,1),(1,128),(2,512),(1,2048)):
        x=torch.randn(b,s,16,128,device='cuda',dtype=dtype).transpose(1,2)
        c=torch.randn(b,s,128,device='cuda',dtype=dtype)
        sn=torch.randn_like(c)
        expected=x*c[:,None]+torch.cat((-x[...,64:],x[...,:64]),-1)*sn[:,None]
        variants={}
        def old():
            y=torch.empty(x.shape,device=x.device,dtype=x.dtype)
            _rotary[(triton.cdiv(x.numel(),512),)](x,c,sn,y,16,s,128,*x.stride(),x.numel(),512,enable_fp_fusion=False)
            return y
        variants['old']=old
        for block in (128,256,512,1024,2048):
            for warps in (4,8):
                variants[f'block{block}_w{warps}']=lambda block=block,warps=warps:rope_rows(x,c,sn,block=block,num_warps=warps)
        records+=run_case(variants,expected,dict(op='rope',dtype=str(dtype),b=b,seq=s))
    return records


def gemm_cases(dtype):
    records=[]
    g=torch.randn(6144,2048,device='cuda',dtype=dtype)*.02
    u=torch.randn_like(g)*.02
    # Changes are isolated: launch tuning at GROUP_M=0, then order-only at a
    # fixed tile. This does not claim to exhaust the configuration space.
    tiles=[(16,32,64,4,2),(16,64,64,4,3),(32,64,32,4,3),(32,64,64,4,3),
           (32,64,128,4,3),(32,128,64,4,3),(64,64,64,4,3),(64,128,32,8,3),
           (32,64,64,8,3),(32,64,64,4,2),(32,64,64,4,4)]
    for m in (1,16,128,512,1024):
        x=torch.randn(m,2048,device='cuda',dtype=dtype)
        expected=F.silu(F.linear(x,g))*F.linear(x,u)
        def cublas():
            a,b=F.linear(x,g),F.linear(x,u)
            y=torch.empty_like(a)
            _activation[(triton.cdiv(y.numel(),512),)](a,b,y,y.numel(),512,enable_fp_fusion=False)
            return y
        variants={'cublas':cublas,'original_autotune':lambda:gate_up_swiglu_triton(x,g,u)}
        for bm,bn,bk,w,st in tiles:
            variants[f'tile{bm}_{bn}_{bk}_w{w}_st{st}']=lambda bm=bm,bn=bn,bk=bk,w=w,st=st:gate_up_tuned(
                x,g,u,bm=bm,bn=bn,bk=bk,warps=w,stages=st)
        for group in (1,4,8):
            variants[f'order_g{group}']=lambda group=group:gate_up_tuned(x,g,u,bm=32,bn=64,bk=64,group_m=group)
        records+=run_case(variants,expected,dict(op='gemm',dtype=str(dtype),m=m,n=6144,k=2048))
    return records


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('section',choices=['decode','rows','gemm'])
    args=parser.parse_args()
    torch.manual_seed(20260928)
    root=Path(__file__).resolve().parents[1]
    sources=['ops/triton/decode_split_kv.py','ops/triton/rowwise_tuning.py',
             'ops/triton/gemm_tuning.py','ops/triton/gate_up_swiglu.py','benchmarks/benchmark_kernel_tuning.py']
    report=dict(manifest=dict(torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),seed=20260928,
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        timing='9 alternating CUDA Graph samples, 10 calls/replay; includes merge/copies; warm cache',
        source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in sources}),records=[])
    target=root/'artifacts'/f'kernel_tuning_{args.section}.json'
    for dtype in (torch.float16,torch.bfloat16):
        report['records']+=dict(decode=decode_cases,rows=row_cases,gemm=gemm_cases)[args.section](dtype)
        target.write_text(json.dumps(report,indent=2)+'\n')
    print(target,flush=True)


if __name__=='__main__': main()
