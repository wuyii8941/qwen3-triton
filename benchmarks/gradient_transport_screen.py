"""Micro-screens only: not full-model adoption or FlashOptim replacement."""
import json,time,statistics
from pathlib import Path
import torch,triton
import triton.language as tl


@triton.jit
def multi_adam(P,G,M,V,L,T,COUNT:tl.constexpr,LR,B1,B2,C1,C2,BC1,BC2,BLOCK:tl.constexpr):
    tile=tl.program_id(0)
    lo=0;hi=COUNT
    while lo+1<hi:
        mid=(lo+hi)//2
        boundary=tl.load(T+mid)
        right=tile>=boundary
        lo=tl.where(right,mid,lo);hi=tl.where(right,hi,mid)
    base=tl.load(T+lo);n=tl.load(L+lo)
    i=(tile-base)*BLOCK+tl.arange(0,BLOCK);mask=i<n
    p=tl.load(P+lo).to(tl.pointer_type(tl.bfloat16))
    g=tl.load(G+lo).to(tl.pointer_type(tl.bfloat16))
    m=tl.load(M+lo).to(tl.pointer_type(tl.bfloat16))
    v=tl.load(V+lo).to(tl.pointer_type(tl.bfloat16))
    pv=tl.load(p+i,mask,0).to(tl.float32);gv=tl.load(g+i,mask,0).to(tl.float32)
    mv=B1*tl.load(m+i,mask,0).to(tl.float32)+C1*gv
    vv=B2*tl.load(v+i,mask,0).to(tl.float32)+C2*gv*gv
    pv=pv-LR*.01*pv
    denom=tl.div_rn(tl.sqrt(vv),BC2)+1e-8
    pv=pv-tl.div_rn(tl.div_rn(LR,BC1)*mv,denom)
    tl.store(p+i,pv,mask);tl.store(m+i,mv,mask);tl.store(v+i,vv,mask)


def error(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))


def bench(fn):
    for _ in range(3):fn()
    samples=[];peaks=[]
    for _ in range(15):
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();fn();torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
        peaks.append(torch.cuda.max_memory_allocated()-base)
    return dict(median_ms=statistics.median(samples),peak_delta_bytes=max(peaks),samples=samples)


def embedding():
    vocab,dim,n=151936,2048,512
    weight=torch.zeros(vocab,dim,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    dense=torch.randn_like(weight)*.001
    ids=torch.randint(0,256,(n,),device='cuda');upstream=torch.randn(n,dim,device='cuda',dtype=torch.bfloat16)
    def baseline():
        y=torch.nn.functional.embedding(ids,weight)
        return dense+torch.autograd.grad(y,weight,upstream)[0]
    def candidate():
        # Include unique/grouping and clone cost. Do not overwrite caller's gradient.
        unique,inverse=torch.unique(ids,return_inverse=True)
        sums=torch.zeros(unique.numel(),dim,device='cuda',dtype=torch.float32)
        sums.index_add_(0,inverse,upstream.float())
        out=dense.clone();out.index_copy_(0,unique,dense[unique]+sums.to(dense.dtype))
        return out
    rel=error(candidate(),baseline())
    return dict(relative_l2=rel,passed=rel<=.02,baseline=bench(baseline),candidate=bench(candidate),
        scope='Synthetic tied-weight gradient composition; duplicate IDs, row sum FP32, then BF16 cast; no full-model validation.')


def dx():
    a=torch.randn(512,6144,device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a)
    wg=torch.randn(6144,2048,device='cuda',dtype=torch.bfloat16);wu=torch.randn_like(wg)
    def baseline():return a@wg+b@wu
    def candidate():return torch.addmm(a@wg,b,wu,beta=1)
    rel=error(candidate(),baseline())
    return dict(relative_l2=rel,passed=rel<=.02,baseline=bench(baseline),candidate=bench(candidate),
                scope='Unchanged GEMM dimensions, addmm accumulation; BF16 rounding boundary differs.')


def optimizer():
    from training.triton_adamw import TritonAdamW
    sizes=[2048]*64+[2048*2048]*8
    initial=[torch.randn(n,device='cuda',dtype=torch.bfloat16)*.02 for n in sizes]
    grad=[torch.randn_like(p)*.001 for p in initial]
    ref=[torch.nn.Parameter(p.clone()) for p in initial];opt=TritonAdamW(ref,math_mode='rounded')
    p=[x.clone() for x in initial];m=[torch.zeros_like(x) for x in p];v=[torch.zeros_like(x) for x in p]
    def ptr(xs):return torch.tensor([x.data_ptr() for x in xs],device='cuda',dtype=torch.int64)
    pointers=[ptr(xs) for xs in (p,grad,m,v)]
    length=torch.tensor(sizes,device='cuda',dtype=torch.int64);prefix=[0]
    for n in sizes:prefix.append(prefix[-1]+triton.cdiv(n,2048))
    tiles=torch.tensor(prefix,device='cuda',dtype=torch.int64);step=0
    def baseline():
        for x,g in zip(ref,grad):x.grad=g
        opt.step()
    def candidate():
        nonlocal step;step+=1
        # Charge uploading grad pointer metadata; no gradient concatenation.
        pointers[1].copy_(torch.tensor([x.data_ptr() for x in grad],dtype=torch.int64))
        multi_adam[(prefix[-1],)](*pointers,length,tiles,len(p),1e-5,.9,.999,.1,.001,
            1-.9**step,(1-.999**step)**.5,2048,enable_fp_fusion=False)
    for _ in range(4):baseline();candidate()
    rel=error(torch.cat(p),torch.cat([x.detach() for x in ref]))
    moment_error=max(error(x,opt.state[y]['exp_avg']) for x,y in zip(m,ref))
    return dict(relative_l2=rel,momentum_relative_l2=moment_error,passed=max(rel,moment_error)<=.001,
        baseline=bench(baseline),candidate=bench(candidate),
        scope='Pointer-table BF16 unquantized AdamW prototype vs existing TritonAdamW; NOT quantized FlashOptim. No model integration.')


def main():
    root=Path('artifacts/gradient_transport_screen_v1');root.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(20261105)
    for name,fn in [('embedding',embedding),('dx',dx),('optimizer',optimizer)]:
        print('START',name,flush=True)
        try:result=fn()
        except Exception as exc:result=dict(error=str(exc),passed=False)
        (root/f'{name}.json').write_text(json.dumps(result,indent=2))
        print('END',name,json.dumps(result)[:600],flush=True)
        torch.cuda.empty_cache()

if __name__=='__main__':main()
