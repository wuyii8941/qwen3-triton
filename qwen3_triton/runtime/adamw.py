"""FP32 Triton AdamW for local Tensor/DTensor shards, with no native update.

Python Optimizer provides parameter-group/state bookkeeping only. Step counters,
bias corrections, moments, decoupled decay and parameter updates run in Triton.
Native Torch2.10 fused AdamW's double coefficient arithmetic is retained. AMP
GradScaler, sparse, complex, BF16 master/state and higher-order modes are not
silently delegated. This path is not qualified until real tests have passed.
"""
import math
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from .memory import raw_empty,zeros,copy_


@triton.jit
def _adamw_prepare(META,COEFF,ROWS:tl.constexpr,LR:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0)*B+tl.arange(0,B);valid=r<ROWS
    address=tl.load(META+r*7+4,valid,0).to(tl.pointer_type(tl.float32))
    step=tl.load(address,valid,0)+1.
    tl.store(address,step,valid)
    one=tl.full((B,),1.,tl.float64)
    b1=tl.full((B,),B1,tl.float64);b2=tl.full((B,),B2,tl.float64)
    bc1=(one-libdevice.pow(b1,step.to(tl.float64))).to(tl.float32)
    bc2=libdevice.sqrt(one-libdevice.pow(b2,step.to(tl.float64))).to(tl.float32)
    size=(tl.full((B,),LR,tl.float64)/bc1.to(tl.float64)).to(tl.float32)
    tl.store(COEFF+r*3,bc1,valid);tl.store(COEFF+r*3+1,bc2,valid);tl.store(COEFF+r*3+2,size,valid)


@triton.jit
def _adamw_update(META,COEFF,ROWS:tl.constexpr,LR:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,
                  DECAY:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr):
    block=tl.program_id(0);low=0;high=ROWS
    while low+1<high:
        mid=(low+high)//2
        start=tl.load(META+mid*7+6)
        if start<=block:low=mid
        else:high=mid
    r=low;start=tl.load(META+r*7+6);n=tl.load(META+r*7+5)
    pptr=tl.load(META+r*7).to(tl.pointer_type(tl.float32))
    gptr=tl.load(META+r*7+1).to(tl.pointer_type(tl.float32))
    mptr=tl.load(META+r*7+2).to(tl.pointer_type(tl.float32))
    vptr=tl.load(META+r*7+3).to(tl.pointer_type(tl.float32))
    i=(block-start)*B+tl.arange(0,B);valid=i<n
    p=tl.load(pptr+i,valid,0);g=tl.load(gptr+i,valid,0)
    m=tl.load(mptr+i,valid,0);v=tl.load(vptr+i,valid,0)
    b1=tl.full((),B1,tl.float64);b2=tl.full((),B2,tl.float64)
    lr=tl.full((),LR,tl.float64);decay=tl.full((),DECAY,tl.float64);eps=tl.full((),EPS,tl.float64)
    p=(p.to(tl.float64)-lr*decay*p.to(tl.float64)).to(tl.float32)
    m=(b1*m.to(tl.float64)+(1.-b1)*g.to(tl.float64)).to(tl.float32)
    v=(b2*v.to(tl.float64)+(1.-b2)*g.to(tl.float64)*g.to(tl.float64)).to(tl.float32)
    bc2=tl.load(COEFF+r*3+1);size=tl.load(COEFF+r*3+2)
    root=libdevice.sqrt_rn(v)
    denom=(tl.div_rn(root,bc2).to(tl.float64)+eps).to(tl.float32)
    p=p-tl.div_rn(size*m,denom)
    tl.store(pptr+i,p,valid);tl.store(mptr+i,m,valid);tl.store(vptr+i,v,valid)


def local(tensor):
    return tensor.to_local() if hasattr(tensor,'to_local') else tensor


class TritonAdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,*,block=4096):
        if not 0<=lr or not 0<=eps or not 0<=weight_decay or not all(0<=b<1 for b in betas):
            raise ValueError('Invalid AdamW scalar configuration')
        if block not in (1024,2048,4096):raise ValueError('Explicit supported block required')
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
        self.block=block;self.kernel_steps=0

    def load_state_dict(self,state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            if any(group.get(key,False) for key in ['amsgrad','maximize','differentiable','capturable']):
                raise ValueError('Unsupported optimizer checkpoint mode')
            for parameter in group['params']:
                state=self.state.get(parameter)
                if not state:continue
                device=local(parameter).device
                for key in ['step','exp_avg','exp_avg_sq']:
                    source=local(state[key])
                    if source.dtype!=torch.float32:raise ValueError('FP32 optimizer checkpoint state required')
                    target=raw_empty(source.shape,device=device,dtype=torch.float32)
                    if source.is_cuda:copy_(target,source)
                    else:target.copy_(source)  # host checkpoint transfer is DMA
                    state[key]=target

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        if getattr(self,'grad_scale',None) is not None or getattr(self,'found_inf',None) is not None:
            raise ValueError('GradScaler is not supported by this explicit BF16/FP32 path')
        for group in self.param_groups:
            rows=[];blocks=0;device=None
            for parameter in group['params']:
                if parameter.grad is None:continue
                p=local(parameter);g=local(parameter.grad)
                if p.dtype!=torch.float32 or g.dtype!=torch.float32 or not p.is_cuda or p.device!=g.device or not p.is_contiguous() or not g.is_contiguous() or p.shape!=g.shape or g.is_sparse:
                    raise ValueError('Matching contiguous local CUDA FP32 parameter/gradient required')
                if device is not None and device!=p.device:raise ValueError('One CUDA device per group')
                device=p.device
                if not p.numel():continue
                state=self.state[parameter]
                if not state:
                    state['step']=zeros((),device=device,dtype=torch.float32)
                    state['exp_avg']=zeros(p.shape,device=device,dtype=torch.float32)
                    state['exp_avg_sq']=zeros(p.shape,device=device,dtype=torch.float32)
                m,v=local(state['exp_avg']),local(state['exp_avg_sq']);step=state['step']
                if m.shape!=p.shape or v.shape!=p.shape or m.dtype!=torch.float32 or v.dtype!=torch.float32 or step.dtype!=torch.float32 or step.numel()!=1 or step.device!=device:
                    raise ValueError('Invalid local FP32 AdamW state')
                rows.append([p.data_ptr(),g.data_ptr(),m.data_ptr(),v.data_ptr(),step.data_ptr(),p.numel(),blocks])
                blocks+=triton.cdiv(p.numel(),self.block)
            if not rows:continue
            # Pointer/shape metadata originates on the host. Transfer is DMA,
            # not tensor arithmetic; GPU values are not copied to the CPU.
            cpu=torch.tensor(rows,dtype=torch.int64)
            meta=raw_empty(cpu.shape,device=device,dtype=cpu.dtype);meta.copy_(cpu)
            coeff=raw_empty((len(rows),3),device=device,dtype=torch.float32)
            lr=group['lr'];b1,b2=group['betas']
            _adamw_prepare[(triton.cdiv(len(rows),128),)](meta,coeff,len(rows),lr,b1,b2,128,enable_fp_fusion=True)
            _adamw_update[(blocks,)](meta,coeff,len(rows),lr,b1,b2,group['weight_decay'],group['eps'],self.block,num_warps=4,enable_fp_fusion=True)
        self.kernel_steps+=1
        return loss
