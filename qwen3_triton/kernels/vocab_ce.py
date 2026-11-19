"""Vocabulary-local cross-entropy and FP32 gradient accumulation.

Model arithmetic is Triton; collectives live in vocab_parallel.py. The loss
keeps the BF16 projection rounding boundary. No quantization or sampled loss.
The vocabulary-parallel softmax algorithm is established (e.g. Megatron-LM),
not a novel algorithm. This implementation is independently written here.
"""
import torch
import triton
import triton.language as tl
from .loss import _partials


@triton.jit
def _local_rows(X,P,LABELS,LSE,TARGET,S:tl.constexpr,V:tl.constexpr,
                START:tl.constexpr,PARTS:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0);physical=(row//(S-1))*S+row%(S-1)
    c=tl.arange(0,D);p=tl.load(P+row*PARTS+c,c<PARTS,-float('inf'))
    maximum=tl.max(p,0);lse=maximum+tl.log(tl.sum(tl.exp(p-maximum),0))
    label=tl.load(LABELS+physical+1);owned=(label>=START)&(label<START+V)
    target=tl.load(X+physical*V+label-START,owned,0).to(tl.float32)
    tl.store(LSE+row,lse);tl.store(TARGET+row,target)


@triton.jit
def _pack(LSE,MAX,TARGET,PACK,N:tl.constexpr,D:tl.constexpr):
    i=tl.program_id(0)*D+tl.arange(0,D);valid=i<N
    local=tl.load(LSE+i,valid,0);maximum=tl.load(MAX+i,valid,0)
    target=tl.load(TARGET+i,valid,0)
    tl.store(PACK+i,tl.exp(local-maximum),valid);tl.store(PACK+N+i,target,valid)


@triton.jit
def _finish(MAX,PACK,LABELS,LSE,LOSSES,N:tl.constexpr,S:tl.constexpr,
            IGNORE:tl.constexpr,D:tl.constexpr):
    i=tl.program_id(0)*D+tl.arange(0,D);valid=i<N
    lse=tl.load(MAX+i,valid,0)+tl.log(tl.load(PACK+i,valid,1))
    physical=(i//(S-1))*S+i%(S-1)
    label=tl.load(LABELS+physical+1,valid,IGNORE)
    target=tl.load(PACK+N+i,valid,0)
    tl.store(LSE+i,lse,valid);tl.store(LOSSES+i,tl.where(label==IGNORE,0.,lse-target),valid)


@triton.jit
def _means(LOSSES,LABELS,COUNTS,MEANS,R:tl.constexpr,S:tl.constexpr,
           IGNORE:tl.constexpr,D:tl.constexpr):
    source=tl.program_id(0);i=tl.arange(0,D);n=R//S*(S-1)
    physical=source*R+(i//(S-1))*S+i%(S-1)
    label=tl.load(LABELS+physical+1,i<n,IGNORE)
    count=tl.sum(((i<n)&(label!=IGNORE)).to(tl.int32),0)
    tl.device_assert(count>0,'Each data rank needs at least one valid target')
    total=tl.sum(tl.load(LOSSES+source*n+i,i<n,0),0)
    tl.store(COUNTS+source,count);tl.store(MEANS+source,total/count)


@triton.jit
def _grad(X,LABELS,LSE,COUNTS,UP,DZ,R:tl.constexpr,S:tl.constexpr,
          V:tl.constexpr,START:tl.constexpr,IGNORE:tl.constexpr,D:tl.constexpr):
    physical=tl.program_id(0);col=tl.program_id(1)*D+tl.arange(0,D)
    valid_row=physical%S<S-1;source=physical//R
    row=(physical//S)*(S-1)+physical%S
    label=tl.load(LABELS+physical+1,valid_row,IGNORE)
    value=tl.load(X+physical*V+col,(col<V)&valid_row,0).to(tl.float32)
    lse=tl.load(LSE+row,valid_row,0)
    scale=tl.load(UP+source).to(tl.float32)/tl.load(COUNTS+source).to(tl.float32)
    dz=(tl.exp(value-lse)-(col+START==label).to(tl.float32))*scale
    tl.store(DZ+physical*V+col,tl.where(valid_row&(label!=IGNORE),dz,0.),col<V)


@triton.jit
def _accum(PARTIAL,OUT,N:tl.constexpr,WORLD:tl.constexpr,FIRST:tl.constexpr,D:tl.constexpr):
    i=tl.program_id(0)*D+tl.arange(0,D);valid=i<N
    value=tl.load(PARTIAL+i,valid,0).to(tl.float32)*(1./WORLD)
    if not FIRST:value=tl.load(OUT+i,valid,0)+value
    tl.store(OUT+i,value,valid)


def local_statistics(logits,labels,sequence,start):
    n=labels.numel()//sequence*(sequence-1);v=logits.shape[1]
    block=2048;parts=triton.cdiv(v,block)
    p=torch.empty((n,parts),device=logits.device,dtype=torch.float32)
    lse=torch.empty(n,device=logits.device,dtype=torch.float32);target=torch.empty_like(lse)
    _partials[(n,parts)](logits,p,sequence,v,parts,block,enable_fp_fusion=False)
    _local_rows[(n,)](logits,p,labels,lse,target,sequence,v,start,parts,
                     triton.next_power_of_2(parts),enable_fp_fusion=False)
    return lse,target


def packed_statistics(local_lse,maximum,target):
    pack=torch.empty((2,local_lse.numel()),device=local_lse.device,dtype=torch.float32)
    _pack[(triton.cdiv(local_lse.numel(),256),)](local_lse,maximum,target,pack,local_lse.numel(),256,enable_fp_fusion=False)
    return pack


def finish_statistics(maximum,pack,labels,rows_per_rank,sequence,world,ignore):
    n=maximum.numel();lse=torch.empty_like(maximum);losses=torch.empty_like(maximum)
    counts=torch.empty(world,device=maximum.device,dtype=torch.int32)
    means=torch.empty(world,device=maximum.device,dtype=torch.float32)
    _finish[(triton.cdiv(n,256),)](maximum,pack,labels,lse,losses,n,sequence,ignore,256,enable_fp_fusion=False)
    _means[(world,)](losses,labels,counts,means,rows_per_rank,sequence,ignore,
                    triton.next_power_of_2(rows_per_rank//sequence*(sequence-1)),debug=True,enable_fp_fusion=False)
    return lse,counts,means


def logit_gradient(logits,labels,lse,counts,up,rows_per_rank,sequence,start,ignore):
    # Separate allocation preserves saved logits and repeated backward semantics.
    dz=torch.empty_like(logits)
    _grad[(logits.shape[0],triton.cdiv(logits.shape[1],2048))](
        logits,labels,lse,counts,up,dz,rows_per_rank,sequence,logits.shape[1],start,ignore,2048,enable_fp_fusion=False)
    return dz


def accumulate_weight_gradient(partial,out,world,first):
    _accum[(triton.cdiv(out.numel(),1024),)](partial,out,out.numel(),world,first,1024,enable_fp_fusion=False)
