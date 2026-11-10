import pytest
import torch
import torch.nn.functional as F
from ops.triton.linear_decode import linear_decode
from ops.triton.weight_only_gemv import quantize_groupwise

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('batch,k',[(1,128),(2,2048),(4,256)])
def test_small_m_linear(dtype,batch,k):
    torch.manual_seed(887)
    with torch.inference_mode():
        x=torch.randn(batch,k,device='cuda',dtype=dtype)
        w=torch.randn(129,k,device='cuda',dtype=dtype)*.02
        snapshots=[w]+[quantize_groupwise(w,g) for g in (16,32,64)]
        for packed in snapshots:
            oracle=F.linear(x,packed if isinstance(packed,torch.Tensor) else packed.dequantize())
            for bn in (1,2,4,8):
                for shared in (False,True):
                    out=linear_decode(x,packed,block_n=bn,share_batch=shared)
                    torch.testing.assert_close(out,oracle,atol=.005,rtol=.02)


def test_linear_contract():
    x=torch.ones(1,128,device='cuda',dtype=torch.float16)
    w=torch.ones(129,128,device='cuda',dtype=x.dtype)
    with pytest.raises(ValueError,match='Inference'):
        linear_decode(x,w)
    with torch.inference_mode():
        with pytest.raises(ValueError,match='Contiguous'):
            linear_decode(torch.ones(1,256,device='cuda',dtype=x.dtype)[:,::2],w)


def test_large_vocabulary_grid():
    with torch.inference_mode():
        x=torch.ones(1,128,device='cuda',dtype=torch.float16)
        w=torch.ones(65537,128,device='cuda',dtype=x.dtype)*.01
        y=linear_decode(x,w,block_n=1)
        torch.testing.assert_close(y,F.linear(x,w),atol=.001,rtol=.001)


def test_tied_head_snapshot_and_restore():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from model.hf_inference import qwen3_triton_inference
    from model.static_decode import StaticDecodeRunner
    c=Qwen3Config(vocab_size=256,hidden_size=256,intermediate_size=512,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=64,
        tie_word_embeddings=True)
    c._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(c).cuda().half().eval()
    ids=torch.randint(256,(2,10),device='cuda')
    with torch.inference_mode():
        original=model.lm_head.weight.clone()
        ref=model(ids,use_cache=False,logits_to_keep=1).logits
        with qwen3_triton_inference(model,fuse_qk=True,compact_gqa=True,static_decode=True,
                                    lm_head_impl='int8',lm_head_group=32):
            r=StaticDecodeRunner(model,2,16);r.prefill(ids[:,:9]);r.capture(ids[:,9:])
            out=r.decode(ids[:,9:])
            assert (out.float()-ref.float()).norm()/ref.float().norm()<.02
            assert r.quantized_refs
        assert model.lm_head.weight is model.model.embed_tokens.weight
        assert not hasattr(model.lm_head,'_int8_lm_head')
        assert not any('forward' in m.__dict__ for m in model.modules())
        torch.testing.assert_close(model.lm_head.weight,original,atol=0,rtol=0)


def test_native_contiguous_head():
    from types import MethodType
    from model.hf_inference import make_lm_head_forward
    with torch.inference_mode():
        head=torch.nn.Linear(256,129,bias=False,device='cuda',dtype=torch.float16)
        x=torch.randn(4,17,256,device='cuda',dtype=torch.float16)[:,-1:]
        assert not x.is_contiguous()
        out=MethodType(make_lm_head_forward('contiguous'),head)(x)
        torch.testing.assert_close(out,head(x),atol=.002,rtol=.01)
