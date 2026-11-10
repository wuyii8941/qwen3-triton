"""Experimental native-formula AdamW, no quantization or master-weight change.

Follows torch 2.5 fused_adam_utils.cuh's formula/order: unrounded FP32 moment
updates feed the parameter update, then moments/parameters cast on store.
NOT bitwise identical: native hyperparameters are double; this candidate uses
FP32 scalar arithmetic. Parameter/moment dtype and optimizer objective stay
unchanged; multi-step and trajectory tolerances must be checked explicitly.
Single parameter group; no AMP scaling, AMSGrad, sparse grads or capture yet.
"""
import torch
import math
import triton
import triton.language as tl


@triton.jit
def _update(P,G,M,V,N:tl.constexpr,LR,B1,B2,C1,C2,WD,EPS,BC1,BC2,
            CORRECTED:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    p=tl.load(P+i,i<N,0).to(tl.float32)
    g=tl.load(G+i,i<N,0).to(tl.float32)
    m=tl.load(M+i,i<N,0).to(tl.float32)
    v=tl.load(V+i,i<N,0).to(tl.float32)
    p=p-LR*WD*p
    if CORRECTED:
        # Do not subtract already-rounded float32 beta from 1 inside kernel.
        m=B1*m+C1*g
        v=B2*v+C2*g*g
        denom=tl.div_rn(tl.sqrt(v),BC2)+EPS
        p=p-tl.div_rn(tl.div_rn(LR,BC1)*m,denom)
    else:
        m=B1*m+(1.-B1)*g
        v=B2*v+(1.-B2)*g*g
        p=p-(LR/BC1)*m/(tl.sqrt(v)/BC2+EPS)
    tl.store(P+i,p,i<N);tl.store(M+i,m,i<N);tl.store(V+i,v,i<N)


class TritonAdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,math_mode='rounded'):
        if math_mode not in ('legacy','rounded'):raise ValueError('Unknown math mode')
        self.math_mode=math_mode
        if (not all(math.isfinite(x) for x in (lr,eps,weight_decay,*betas)) or
                lr<0 or eps<0 or weight_decay<0 or len(betas)!=2 or not all(0<=b<1 for b in betas)):
            raise ValueError('Invalid AdamW hyperparameters')
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,math_mode=math_mode))
        if len(self.param_groups)!=1:raise ValueError('Single group experiment only')
        group=self.param_groups[0]
        if any(group.get(k,False) for k in ('amsgrad','maximize','capturable','differentiable')):
            raise ValueError('Unsupported AdamW group option')
        if len({id(p) for p in group['params']})!=len(group['params']):
            raise ValueError('Duplicate parameters unsupported')

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('Closures unsupported')
        group=self.param_groups[0];b1,b2=group['betas']
        mode=group.get('math_mode',self.math_mode)
        for p in group['params']:
            if p.grad is None:continue
            g=p.grad
            if not p.is_cuda or not p.is_contiguous() or g.is_sparse or not g.is_contiguous():
                raise ValueError('Contiguous dense CUDA parameters/gradients required')
            if p.dtype not in (torch.float16,torch.bfloat16,torch.float32) or g.dtype!=p.dtype:
                raise ValueError('Shared FP16/BF16/FP32 dtype required')
            state=self.state[p]
            if not state:
                state['step']=0;state['exp_avg']=torch.zeros_like(p);state['exp_avg_sq']=torch.zeros_like(p)
            state['step']+=1;t=state['step']
            if p.numel()==0:continue
            _update[(triton.cdiv(p.numel(),2048),)](p,g,state['exp_avg'],state['exp_avg_sq'],p.numel(),
                group['lr'],b1,b2,1-b1,1-b2,group['weight_decay'],group['eps'],1-b1**t,(1-b2**t)**.5,
                mode=='rounded',2048,num_warps=4,enable_fp_fusion=mode=='legacy')
