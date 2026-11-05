"""HF adapter compatibility, noncontiguous RoPE and exception restoration."""
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3 as hf
from model.hf_inference import qwen3_triton_inference, _rope, _norm_forward, _mlp_forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_hf_operator_semantics(dtype):
    torch.manual_seed(42)
    cfg = Qwen3Config(hidden_size=128, intermediate_size=257, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=64)
    norm = hf.Qwen3RMSNorm(128).cuda().to(dtype)
    mlp = hf.Qwen3MLP(cfg).cuda().to(dtype)
    with torch.inference_mode():
        norm.weight.copy_(torch.randn_like(norm.weight))
        x = torch.randn(2, 17, 128, device='cuda', dtype=dtype)
        for a, b in ((_norm_forward(norm, x), norm(x)), (_mlp_forward(mlp, x), mlp(x))):
            torch.testing.assert_close(a, b, atol=0.002, rtol=0.02)
        # HF attention supplies head-transposed views, not contiguous BHSD.
        q = torch.randn(2,17,2,64,device='cuda',dtype=dtype).transpose(1,2)
        k = torch.randn(2,17,1,64,device='cuda',dtype=dtype).transpose(1,2)
        angles = torch.randn(2,17,32,device='cuda').repeat(1,1,2)
        c,s = angles.cos().to(dtype), angles.sin().to(dtype)
        for a,b in zip(_rope(q,k,c,s), hf.apply_rotary_pos_emb(q,k,c,s)):
            torch.testing.assert_close(a,b,atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('fuse_residual', [False, True])
@pytest.mark.parametrize('decode_mlp', [False, True])
@pytest.mark.parametrize('multirow_norm', [False,True])
def test_adapter_restores_after_exception(fuse_residual,decode_mlp,multirow_norm):
    cfg = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=64)
    model = Qwen3ForCausalLM(cfg).cuda().half().eval()
    original = hf.apply_rotary_pos_emb
    ids = torch.tensor([[1,2,3]], device='cuda')
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits
        with pytest.raises(RuntimeError, match='intentional'):
            with qwen3_triton_inference(model, fuse_residual=fuse_residual,decode_mlp=decode_mlp,multirow_norm=multirow_norm):
                actual = model(ids, use_cache=False).logits
                torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
                raise RuntimeError('intentional')
        assert hf.apply_rotary_pos_emb is original
        assert not any('forward' in m.__dict__ for m in model.modules())
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match='inference_mode'):
        with qwen3_triton_inference(model):
            pass
