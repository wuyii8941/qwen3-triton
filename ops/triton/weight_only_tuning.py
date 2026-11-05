"""Experimental W8A16 implementations: explicit scale reuse and Tensor Cores."""
import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemv(X,G,U,SG,SU,Y,N:tl.constexpr,K:tl.constexpr,GROUP:tl.constexpr,
                  BN:tl.constexpr,KG:tl.constexpr):
    m=tl.program_id(0)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    kg=tl.arange(0,KG)
    j=tl.arange(0,GROUP)
    k=kg[:,None]*GROUP+j[None,:]
    x=tl.load(X+m*K+k,k<K,0).to(tl.float32)
    mask=(n[:,None,None]<N)&(k[None,:,:]<K)
    g=tl.load(G+n[:,None,None]*K+k[None,:,:],mask,0).to(tl.float32)
    u=tl.load(U+n[:,None,None]*K+k[None,:,:],mask,0).to(tl.float32)
    # Load each group scale once, broadcast explicitly over its coefficients.
    gs=tl.load(SG+n[:,None]*(K//GROUP)+kg[None,:],(n[:,None]<N)&(kg[None,:]<K//GROUP),0)
    us=tl.load(SU+n[:,None]*(K//GROUP)+kg[None,:],(n[:,None]<N)&(kg[None,:]<K//GROUP),0)
    g=(g*gs[:,:,None]).to(X.dtype.element_ty).to(tl.float32)
    u=(u*us[:,:,None]).to(X.dtype.element_ty).to(tl.float32)
    # Flatten back to [output rows, K] before reducing. Nested 3-D reductions
    # produced incorrect rows for BN=2/4 on this pinned Triton 3.1 toolchain.
    g=tl.reshape(g,(BN,KG*GROUP))
    u=tl.reshape(u,(BN,KG*GROUP))
    x=tl.reshape(x,(KG*GROUP,))
    a=tl.sum(g*x[None,:],1).to(Y.dtype.element_ty).to(tl.float32)
    b=tl.sum(u*x[None,:],1).to(Y.dtype.element_ty).to(tl.float32)
    silu=(a/(1+tl.exp(-a))).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+m*N+n,silu*b,n<N)


@triton.jit
def _w8_tensorcore(X,G,U,SG,SU,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                    GROUP:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    ag=tl.zeros((BM,BN),tl.float32)
    au=tl.zeros((BM,BN),tl.float32)
    for tile in range(tl.cdiv(K,BK)):
        k=tile*BK+kk
        x=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0)
        mask=(n[None,:]<N)&(k[:,None]<K)
        g=tl.load(G+n[None,:]*K+k[:,None],mask,0).to(tl.float32)
        u=tl.load(U+n[None,:]*K+k[:,None],mask,0).to(tl.float32)
        gs=tl.load(SG+n[None,:]*(K//GROUP)+k[:,None]//GROUP,mask,0)
        us=tl.load(SU+n[None,:]*(K//GROUP)+k[:,None]//GROUP,mask,0)
        g=(g*gs).to(x.dtype)
        u=(u*us).to(x.dtype)
        ag+=tl.dot(x,g)
        au+=tl.dot(x,u)
    a=ag.to(Y.dtype.element_ty).to(tl.float32)
    b=au.to(Y.dtype.element_ty).to(tl.float32)
    silu=(a/(1+tl.exp(-a))).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+m[:,None]*N+n[None,:],silu*b,(m[:,None]<M)&(n[None,:]<N))


def gate_up_int8_tuned(x,gate,up,*,impl='grouped',bn=1,bk=64,warps=4,stages=3):
    """Internal tuning entry; public contract established by caller/tests."""
    m,k=x.shape
    n=gate.values.shape[0]
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    args=(x,gate.values,up.values,gate.scales,up.scales,out)
    if impl=='grouped':
        kernel=_grouped_gemv[(m,triton.cdiv(n,bn))](*args,n,k,gate.group_size,
            bn,triton.next_power_of_2(k//gate.group_size),num_warps=warps,enable_fp_fusion=False)
    elif impl=='tensorcore':
        kernel=_w8_tensorcore[(triton.cdiv(m,16),triton.cdiv(n,bn))](*args,m,n,k,
            gate.group_size,16,bn,bk,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    else:
        raise ValueError('Unknown implementation')
    return out
