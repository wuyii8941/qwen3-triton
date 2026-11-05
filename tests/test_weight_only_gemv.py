import pytest
import torch
import torch.nn.functional as F
from ops.triton.weight_only_gemv import quantize_groupwise, gate_up_int8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('group', [16, 32, 64, 128])
def test_weight_only_against_dequantized_oracle(dtype, group):
    torch.manual_seed(847)
    with torch.inference_mode():
        x = torch.randn(2, 256, device='cuda', dtype=dtype)
        g = torch.randn(129, 256, device='cuda', dtype=dtype) * .02
        u = torch.randn_like(g) * .02
        g[0].zero_()
        g0, u0 = g.clone(), u.clone()
        qg, qu = quantize_groupwise(g, group), quantize_groupwise(u, group)
        torch.testing.assert_close(g, g0, atol=0, rtol=0)
        torch.testing.assert_close(u, u0, atol=0, rtol=0)
        expected = F.silu(F.linear(x, qg.dequantize())) * F.linear(x, qu.dequantize())
        for bn in (1, 2, 4):
            for impl in ('naive','grouped'):
                actual = gate_up_int8(x, qg, qu, block_n=bn,impl=impl)
                torch.testing.assert_close(actual, expected, atol=.003, rtol=.02)
                assert actual.dtype == dtype
        assert (qg.values[0] == 0).all()
        assert torch.isfinite(qg.scales).all()


def test_model_quantization_scope_restore_and_batch_dispatch():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from model.hf_inference import qwen3_triton_inference,quantize_decode_mlp
    from model.static_decode import StaticDecodeRunner
    c=Qwen3Config(vocab_size=128,hidden_size=2048,intermediate_size=6144,
        num_hidden_layers=1,num_attention_heads=16,num_key_value_heads=8,head_dim=128)
    c._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(c).cuda().half().eval()
    flags=dict(decode_mlp=True,fuse_qk=True,compact_gqa=True,static_decode=True,
               direct_k_cache=True,direct_v_cache=True)
    torch.manual_seed(993)
    ids=torch.randint(128,(2,9),device='cuda')
    with torch.inference_mode():
        original=model.model.layers[0].mlp.gate_proj.weight.clone()
        with qwen3_triton_inference(model,**flags):
            r=StaticDecodeRunner(model,2,12);r.prefill(ids[:,:8])
            expected=r.decode(ids[:,8:],replay=False).clone()
        packed=quantize_decode_mlp(model,16)
        with qwen3_triton_inference(model,**flags,decode_int8_group=16,
                quantized_mlp=packed,decode_int8_impl='grouped',decode_int8_max_batch=1):
            r=StaticDecodeRunner(model,2,12);r.prefill(ids[:,:8]);r.capture(ids[:,8:])
            actual=r.decode(ids[:,8:])
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            one=StaticDecodeRunner(model,1,12);one.prefill(ids[:1,:8]);one.capture(ids[:1,8:])
            one.decode(ids[:1,8:])
            assert one.quantized_refs
        assert not hasattr(model.model.layers[0].mlp,'_int8_gate_up')
        assert not any('forward' in m.__dict__ for m in model.modules())
        torch.testing.assert_close(model.model.layers[0].mlp.gate_proj.weight,original,atol=0,rtol=0)
