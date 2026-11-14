"""SwiGLU with Qwen BF16 cast edges and explicit forward/backward kernels."""
import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu(G,U,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=i<N
    g=tl.load(G+i,mask,0).to(tl.float32);u=tl.load(U+i,mask,0).to(tl.float32)
    silu=(g/(1.+tl.exp(-g))).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+i,silu*u,mask)

@triton.jit
def _backward(G,U,DY,DG,DU,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=i<N
    g=tl.load(G+i,mask,0).to(tl.float32);u=tl.load(U+i,mask,0).to(tl.float32)
    dy=tl.load(DY+i,mask,0).to(tl.float32);sig=1./(1.+tl.exp(-g))
    silu=(g*sig).to(G.dtype.element_ty).to(tl.float32)
    dact=(dy*u).to(G.dtype.element_ty).to(tl.float32)
    tl.store(DG+i,dact*sig*(1.+g*(1.-sig)),mask);tl.store(DU+i,dy*silu,mask)

class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,g,u):
        if not g.is_cuda or g.device!=u.device or g.shape!=u.shape or g.dtype!=u.dtype or not g.is_contiguous() or not u.is_contiguous() or g.numel()<1:
            raise ValueError('Matching contiguous nonempty CUDA inputs required')
        if g.dtype not in (torch.float32,torch.bfloat16,torch.float16):raise TypeError('Floating point required')
        y=torch.empty_like(g);_swiglu[(triton.cdiv(g.numel(),256),)](g,u,y,g.numel(),256,enable_fp_fusion=False)
        ctx.save_for_backward(g,u);return y
    @staticmethod
    def backward(ctx,dy):
        g,u=ctx.saved_tensors;dy=dy.contiguous();dg=torch.empty_like(g);du=torch.empty_like(u)
        _backward[(triton.cdiv(g.numel(),256),)](g,u,dy,dg,du,g.numel(),256,enable_fp_fusion=False)
        return dg,du

def swiglu(gate,up):return _SwiGLU.apply(gate,up)
