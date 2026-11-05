import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3 as hf
from model.hf_inference import qwen3_triton_inference
from ops.triton.hf_qk_norm_rope import hf_qk_norm_rope

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('shape',[(1,1,4,2,64),(2,17,16,8,128),(1,129,6,2,96)])
def test_fused_qk_hf_rounding(dtype,shape):
    b,t,hq,hk,d=shape
    torch.manual_seed(124)
    q=torch.randn(b,t,hq,d,device='cuda',dtype=dtype)
    k=torch.randn(b,t,hk,d,device='cuda',dtype=dtype)
    qn=hf.Qwen3RMSNorm(d,eps=1e-6).cuda().to(dtype)
    kn=hf.Qwen3RMSNorm(d,eps=1e-6).cuda().to(dtype)
    # Deliberately non-symmetric tables; test shared and per-batch tables.
    with torch.inference_mode():
        qn.weight.copy_(torch.randn_like(qn.weight))
        kn.weight.copy_(torch.randn_like(kn.weight))
        for cb in {1,b}:
            c=torch.randn(cb,t,d,device='cuda',dtype=dtype)
            s=torch.randn_like(c)
            expected=hf.apply_rotary_pos_emb(qn(q).transpose(1,2),kn(k).transpose(1,2),c,s)
            actual=hf_qk_norm_rope(q,k,qn.weight,kn.weight,c,s)
            for a,e in zip(actual,expected):
                torch.testing.assert_close(a,e,atol=.003,rtol=.01)
                assert a.is_contiguous()


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('fuse_qk,compact_gqa',[(True,False),(False,True),(True,True)])
def test_attention_cache_and_restore(dtype,fuse_qk,compact_gqa):
    torch.manual_seed(125)
    cfg=Qwen3Config(vocab_size=256,hidden_size=256,intermediate_size=512,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=64,
        rms_norm_eps=1e-6)
    cfg._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(cfg).cuda().to(dtype).eval()
    ids=torch.randint(256,(2,20),device='cuda')
    original=hf.apply_rotary_pos_emb
    with torch.inference_mode():
        expected=model(ids,use_cache=False).logits
        with qwen3_triton_inference(model,fuse_qk=fuse_qk,compact_gqa=compact_gqa):
            actual=model(ids,use_cache=False).logits
            torch.testing.assert_close(actual,expected,atol=.02,rtol=.04)
            out=model(ids[:,:17],use_cache=True)
            cache=out.past_key_values
            for step in range(17,20):
                out=model(ids[:,step:step+1],past_key_values=cache,use_cache=True)
                torch.testing.assert_close(out.logits,actual[:,step:step+1],atol=.02,rtol=.04)
        assert hf.apply_rotary_pos_emb is original
        assert not any('forward' in m.__dict__ for m in model.modules())
        torch.testing.assert_close(model(ids,use_cache=False).logits,expected,atol=0,rtol=0)


def test_mask_and_attention_weights_fallback():
    torch.manual_seed(126)
    cfg=Qwen3Config(vocab_size=128,hidden_size=128,intermediate_size=256,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=64)
    cfg._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(cfg).cuda().half().eval()
    ids=torch.randint(128,(2,9),device='cuda')
    mask=torch.ones_like(ids)
    mask[0,:2]=0
    with torch.inference_mode():
        # Native GQA with a real padding mask (normal SDPA dispatch).
        masked_ref=model(ids,attention_mask=mask,use_cache=False).logits
        with qwen3_triton_inference(model,fuse_qk=True,compact_gqa=True):
            masked=model(ids,attention_mask=mask,use_cache=False).logits
        torch.testing.assert_close(masked,masked_ref,atol=.01,rtol=.02)
        expected=model(ids,attention_mask=mask,use_cache=False,output_attentions=True)
        with qwen3_triton_inference(model,fuse_qk=True,compact_gqa=True):
            actual=model(ids,attention_mask=mask,use_cache=False,output_attentions=True)
        torch.testing.assert_close(actual.logits,expected.logits,atol=.01,rtol=.02)
        torch.testing.assert_close(actual.attentions[0],expected.attentions[0],atol=.002,rtol=.02)
        original=hf.apply_rotary_pos_emb
        with pytest.raises(RuntimeError,match='intentional'):
            with qwen3_triton_inference(model,fuse_qk=True,compact_gqa=True):
                raise RuntimeError('intentional')
        assert hf.apply_rotary_pos_emb is original
        assert not any('forward' in m.__dict__ for m in model.modules())
