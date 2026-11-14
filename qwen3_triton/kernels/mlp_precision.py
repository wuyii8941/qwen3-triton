"""Gate/up GEMMs and SwiGLU in one forward kernel, with full Triton backward.

Gate and up outputs remain saved like the baseline activation, avoiding extra
GEMM recomputation in backward and preserving the same BF16 rounding edges.
"""
import torch
import triton
import triton.language as tl
from qwen3_triton.kernels.linear import matmul
from triton.language.extra.cuda import libdevice
from .math_precision import _hf_swiglu_backward as activation_backward
from qwen3_triton.kernels.residual import residual_add

@triton.jit
def _gate_up(X,WG,WU,G,U,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GROUP:tl.constexpr):
    pid=tl.program_id(0);nm=tl.cdiv(M,BM);nn=tl.cdiv(N,BN)
    if GROUP==1:pm,pn=pid//nn,pid%nn
    else:
        width=GROUP*nn;first=(pid//width)*GROUP;height=tl.minimum(nm-first,GROUP)
        pm=first+(pid%width)%height;pn=(pid%width)//height
    r=pm*BM+tl.arange(0,BM);c=pn*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    gate=tl.zeros((BM,BN),tl.float32);up=tl.zeros((BM,BN),tl.float32)
    for block in range(tl.cdiv(K,BK)):
        k=block*BK+kk
        a=tl.load(X+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),0)
        gw=tl.load(WG+c[None,:]*K+k[:,None],(c[None,:]<N)&(k[:,None]<K),0)
        uw=tl.load(WU+c[None,:]*K+k[:,None],(c[None,:]<N)&(k[:,None]<K),0)
        gate+=tl.dot(a,gw,input_precision='ieee');up+=tl.dot(a,uw,input_precision='ieee')
    gate=gate.to(G.dtype.element_ty).to(tl.float32);up=up.to(U.dtype.element_ty).to(tl.float32)
    silu=(tl.div_rn(gate,1.+libdevice.exp(-gate))).to(Y.dtype.element_ty).to(tl.float32)
    offsets=r[:,None]*N+c[None,:];mask=(r[:,None]<M)&(c[None,:]<N)
    tl.store(G+offsets,gate,mask);tl.store(U+offsets,up,mask);tl.store(Y+offsets,silu*up,mask)

class _GateUp(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,wg,wu,optimized):
        if not x.is_cuda or x.ndim<2 or wg.ndim!=2 or wg.shape!=wu.shape or x.shape[-1]!=wg.shape[1]:raise ValueError('Expected CUDA X[...,K] and matching [N,K] gate/up weights')
        if x.dtype not in (torch.float32,torch.bfloat16,torch.float16) or any(t.device!=x.device or t.dtype!=x.dtype or not t.is_contiguous() for t in (x,wg,wu)):raise ValueError('Matching contiguous floating point CUDA inputs required')
        m=x.numel()//x.shape[-1];k=x.shape[-1];n=wg.shape[0]
        if min(m,k,n)<1:raise ValueError('Nonempty inputs required')
        shape=(*x.shape[:-1],n);g=torch.empty(shape,device=x.device,dtype=x.dtype);u=torch.empty_like(g);y=torch.empty_like(g)
        bm,bn,bk,group=(64,64,32,8) if optimized else (32,64,32,1)
        _gate_up[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](x,wg,wu,g,u,y,m,n,k,bm,bn,bk,group,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,wg,wu,g,u);ctx.optimized=optimized;return y
    @staticmethod
    def backward(ctx,dy):
        x,wg,wu,g,u=ctx.saved_tensors;dy=dy.contiguous();m=x.numel()//x.shape[-1];k=x.shape[-1];n=wg.shape[0]
        dg=torch.empty_like(g);du=torch.empty_like(u)
        activation_backward[(triton.cdiv(g.numel(),256),)](g,u,dy,dg,du,g.numel(),256,True,enable_fp_fusion=False)
        dg=dg.view(m,n);du=du.view(m,n);flat=x.view(m,k)
        dxg=matmul(dg,wg,optimized=ctx.optimized);dxu=matmul(du,wu,optimized=ctx.optimized)
        dx=residual_add(dxg,dxu).view(x.shape)
        dwg=matmul(dg.t(),flat,optimized=ctx.optimized);dwu=matmul(du.t(),flat,optimized=ctx.optimized)
        return dx,dwg,dwu,None

def gate_up_swiglu(x,gate_weight,up_weight,*,optimized=False):
    return _GateUp.apply(x,gate_weight,up_weight,optimized)
