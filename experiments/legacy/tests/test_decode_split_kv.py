import pytest
import torch
from ops.triton.decode_split_kv import attention_decode_split


def oracle(q,k,v,scale=None):
    # FP32 dense oracle only for testing. The candidate never repeats K/V.
    factor=q.shape[1]//k.shape[1]
    k=k.float().repeat_interleave(factor,1)
    v=v.float().repeat_interleave(factor,1)
    s=q.float()@k.transpose(-1,-2)*(128**-.5 if scale is None else scale)
    return s.softmax(-1)@v


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize('shape',[(1,16,8,1),(2,16,8,129),(1,8,2,513),(1,4,4,1025)])
def test_split_and_gqa(dtype,shape):
    b,h,hk,s=shape
    torch.manual_seed(82)
    q=torch.randn(b,h,1,128,device='cuda',dtype=dtype)
    # Nonzero storage offset + extra capacity between heads, like static cache.
    k=torch.randn(b,hk,s+11,128,device='cuda',dtype=dtype)[:,:,3:s+3]
    v=torch.randn(b,hk,s+17,128,device='cuda',dtype=dtype)[:,:,5:s+5]
    expected=oracle(q,k,v,scale=.12)
    with torch.inference_mode():
        for grouped in (False,True):
            for splits in (1,2,8,32):
                out=attention_decode_split(q,k,v,splits=splits,group_heads=grouped,scale=.12)
                torch.testing.assert_close(out.float(),expected,atol=.003,rtol=.01)
                assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_split_pipeline_static_cache():
    from ops.triton.attention_pipeline import CompactGQAAttentionPipeline,ContiguousKVCache
    torch.manual_seed(83)
    with torch.inference_mode():
        pipeline=CompactGQAAttentionPipeline(16,8,decode_impl='split_kv',decode_splits=8)
        cache=ContiguousKVCache(2,8,1024,128,device=torch.device('cuda'),dtype=torch.bfloat16)
        k=torch.randn(2,8,511,128,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        cache.append(k,v)
        for step in range(3):
            q=torch.randn(2,16,1,128,device='cuda',dtype=torch.bfloat16)
            kn=torch.randn(2,8,1,128,device='cuda',dtype=torch.bfloat16)
            vn=torch.randn_like(kn)
            out=pipeline.decode(q,kn,vn,cache)
            kc,vc=cache.view()
            torch.testing.assert_close(out.float(),oracle(q,kc,vc),atol=.003,rtol=.01)
            assert cache.seq_len==512+step
