import pytest
import torch
import torch.nn.functional as F
from ops.triton.decode_gate_up import decode_gate_up_triton


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('shape',[(1,2048,6144),(2,129,257),(16,128,64)])
def test_decode_projection(dtype,shape):
    m,k,n=shape
    torch.manual_seed(47)
    x=torch.randn(m,k,device='cuda',dtype=dtype)
    g=torch.randn(n,k,device='cuda',dtype=dtype)/k**.5
    u=torch.randn_like(g)/k**.5
    with torch.inference_mode():
        ref=F.silu(F.linear(x,g))*F.linear(x,u)
        for bn in (1,2,4,8):
            out=decode_gate_up_triton(x,g,u,block_n=bn)
            torch.testing.assert_close(out,ref,atol=.015 if dtype==torch.bfloat16 else .003,rtol=.03)
            assert torch.linalg.vector_norm((out-ref).float())/torch.linalg.vector_norm(ref.float())<.01
