"""One official optimizer kernel bracketed for an external Nsight profiler."""
import torch
import triton
import triton.language as tl
from flashoptim.optimizers import _triton_adam_kernel,_make_grid


def main():
    torch.manual_seed(7)
    n=151936*2048
    p=torch.randn(n,device='cuda',dtype=torch.bfloat16)*.02
    grad=torch.randn_like(p)*.001
    m=torch.zeros(n,device='cuda',dtype=torch.int8)
    v=torch.zeros(n,device='cuda',dtype=torch.uint8)
    ms=torch.zeros(triton.cdiv(n,32),device='cuda',dtype=torch.float16)
    vs=torch.zeros_like(ms)
    def update(step):
        _triton_adam_kernel[lambda meta:_make_grid(n,meta)](m,ms,v,vs,p,grad,p,n,1e-5,.9,.999,1e-8,1e-7,step,
            GROUP_SIZE=32,DECOUPLED_WEIGHT_DECAY=True,PARAM_DTYPE=tl.bfloat16,USE_ECC=False,
            QUANTIZE_OPTIM_STATES=True,SIGNED_ERROR_T=tl.int8,NUM_MANTISSA_BITS=7,
            SIGNED_MAX_VAL=127,BLOCK_SIZE_N=1024)
    for step in range(1,5):update(step)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    update(5)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__=='__main__':main()
