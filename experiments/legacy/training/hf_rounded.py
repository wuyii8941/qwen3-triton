"""HF-rounded Triton forwards with native-autograd recomputed backwards.

First isolate rounding from hand-written backward math. Not a claim of fully
Triton training: backward intentionally uses the exact native expression.
"""
import torch
from model.hf_inference import _norm_impl,_activation_impl


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,eps):
        ctx.save_for_backward(x,w);ctx.eps=eps
        return _norm_impl(x,w,eps)

    @staticmethod
    def backward(ctx,grad):
        x,w=ctx.saved_tensors
        with torch.enable_grad():
            xx=x.detach().requires_grad_();ww=w.detach().requires_grad_()
            xf=xx.float()
            normalized=(xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+ctx.eps)).to(x.dtype)
            y=ww*normalized
            dx,dw=torch.autograd.grad(y,(xx,ww),grad)
        return dx,dw,None


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,g,u):
        ctx.save_for_backward(g,u)
        return _activation_impl(g,u)

    @staticmethod
    def backward(ctx,grad):
        g,u=ctx.saved_tensors
        with torch.enable_grad():
            gg=g.detach().requires_grad_();uu=u.detach().requires_grad_()
            y=torch.nn.functional.silu(gg)*uu
            dg,du=torch.autograd.grad(y,(gg,uu),grad)
        return dg,du


norm=_Norm.apply
activation=_Activation.apply


class _ActivationTritonBackward(_Activation):
    @staticmethod
    def backward(ctx,grad):
        from ops.triton.hf_swiglu_backward import swiglu_backward
        return swiglu_backward(*ctx.saved_tensors,grad)


activation_triton_backward=_ActivationTritonBackward.apply
