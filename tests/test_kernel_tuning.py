import pytest
import torch
import torch.nn.functional as F
from ops.triton.rowwise_tuning import rms_norm_rows,rope_rows
from ops.triton.gemm_tuning import gate_up_tuned

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('n',[128,129,2048])
def test_multirow_norm(dtype,n):
    torch.manual_seed(90)
    x=torch.randn(33,n,device='cuda',dtype=dtype)
    w=torch.randn(n,device='cuda',dtype=dtype)
    ref=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(dtype)*w
    with torch.inference_mode():
        for rows in (1,2,4,8,16):
            out=rms_norm_rows(x,w,rows=rows)
            torch.testing.assert_close(out,ref,atol=.003,rtol=.01)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_multirow_rope(dtype):
    torch.manual_seed(91)
    x=torch.randn(2,17,4,128,device='cuda',dtype=dtype).transpose(1,2)
    c=torch.randn(2,17,128,device='cuda',dtype=dtype)
    s=torch.randn_like(c)
    ref=x*c[:,None]+torch.cat((-x[...,64:],x[...,:64]),-1)*s[:,None]
    with torch.inference_mode():
        for block in (128,256,512,1024,2048):
            out=rope_rows(x,c,s,block=block)
            torch.testing.assert_close(out,ref,atol=0,rtol=0)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_gemm_grouped_tail(dtype):
    torch.manual_seed(92)
    x=torch.randn(133,129,device='cuda',dtype=dtype)
    g=torch.randn(145,129,device='cuda',dtype=dtype)/129**.5
    u=torch.randn_like(g)/129**.5
    ref=F.silu(F.linear(x,g))*F.linear(x,u)
    with torch.inference_mode():
        for group in (0,1,4,8):
            for bm,bn,bk in ((16,32,32),(32,64,64),(64,128,32)):
                out=gate_up_tuned(x,g,u,bm=bm,bn=bn,bk=bk,group_m=group)
                torch.testing.assert_close(out,ref,atol=.02,rtol=.03)
