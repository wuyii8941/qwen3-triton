import pytest
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import Qwen3Config, Qwen3ForCausalLM
from model.hf_inference import qwen3_triton_inference, pack_qkv_weights
from model.static_decode import StaticDecodeCache, StaticDecodeRunner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def config():
    c = Qwen3Config(vocab_size=256, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, rms_norm_eps=1e-6)
    c._attn_implementation = 'sdpa'
    return c


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('batch', [1, 2])
def test_flash_valid_prefix_and_graph(dtype, batch):
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        cache = StaticDecodeCache(config(), batch, 48, device='cuda', dtype=dtype)
        q = torch.randn(batch, 4, 1, 64, device='cuda', dtype=dtype)
        k = torch.randn(batch, 2, 31, 64, device='cuda', dtype=dtype)
        v = torch.randn_like(k)
        # Poison the unused tail to detect accidental full-capacity attention.
        cache.key_cache[0].fill_(float('nan'))
        cache.value_cache[0].fill_(float('nan'))
        cache.prepare(0, 31)
        cache.update(k, v, 0)
        def fn():
            return cache.attend_decode(q, 0, 64**-.5).transpose(1, 2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn()
        for length in (17, 31, 19):
            cache.used.fill_(length)
            graph.replay()
            expected = torch.nn.functional.scaled_dot_product_attention(
                q, k[:, :, :length], v[:, :, :length], enable_gqa=True)
            torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('packed,direct,direct_v', [(False, False, False), (True, False, False),
    (False, True, False), (True, True, False), (False, True, True), (True, True, True)])
def test_model_incremental_graph_and_restore(dtype, packed, direct, direct_v):
    torch.manual_seed(301)
    model = Qwen3ForCausalLM(config()).cuda().to(dtype).eval()
    ids = torch.randint(256, (2, 24), device='cuda')
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected = model(ids, use_cache=False).logits
        with qwen3_triton_inference(model, fuse_qk=True, compact_gqa=True,
                packed_qkv=packed, static_decode=True, direct_k_cache=direct,direct_v_cache=direct_v):
            runner = StaticDecodeRunner(model, 2, 24)
            out = runner.prefill(ids[:, :17])
            torch.testing.assert_close(out, expected[:, 16:17], atol=.02, rtol=.04)
            runner.capture(ids[:, 17:18])
            # Different positions and tokens, not replaying a single fixed state.
            for pos in range(17, 24):
                out = runner.decode(ids[:, pos:pos+1])
                torch.testing.assert_close(out, expected[:, pos:pos+1], atol=.02, rtol=.04)
            with pytest.raises(ValueError, match='capacity'):
                runner.decode(ids[:, :1])
            # Refill the same storage and reuse graph after a shorter prompt.
            runner.prefill(ids[:, :11])
            for pos in range(11, 15):
                out = runner.decode(ids[:, pos:pos+1], replay=(pos % 2 == 0))
                torch.testing.assert_close(out, expected[:, pos:pos+1], atol=.02, rtol=.04)
        assert '_update_causal_mask' not in model.model.__dict__
        assert not any('forward' in m.__dict__ or '_packed_qkv' in m.__dict__ for m in model.modules())
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, atol=0, rtol=0)


def test_contract_and_exception_restore():
    model = Qwen3ForCausalLM(config()).cuda().half().eval()
    ids = torch.ones((1, 5), device='cuda', dtype=torch.long)
    with torch.inference_mode():
        with pytest.raises(ValueError, match='static_decode=True'):
            StaticDecodeRunner(model, 1, 12).prefill(ids)
        with pytest.raises(ValueError, match='Packed QKV'):
            with qwen3_triton_inference(model, packed_qkv=True): pass
        packed = pack_qkv_weights(model)
        with qwen3_triton_inference(model, fuse_qk=True, compact_gqa=True,
                static_decode=True, packed_qkv=True, packed_weights=packed):
            runner = StaticDecodeRunner(model, 1, 12)
            with pytest.raises(ValueError, match='Prefill'):
                runner.decode(ids[:, :1])
            runner.cache.prepare(0, 5)
            with pytest.raises(ValueError, match='padding'):
                model(ids, attention_mask=torch.ones_like(ids), use_cache=True,
                      past_key_values=runner.cache, cache_position=runner.cache.positions)
            with pytest.raises(ValueError, match='initial prefill'):
                runner.cache.prepare(5, 2)
        assert '_update_causal_mask' not in model.model.__dict__


@pytest.mark.parametrize('length', [1, 15])
def test_short_prefill_and_exception_restore(length):
    model = Qwen3ForCausalLM(config()).cuda().half().eval()
    ids = torch.randint(256, (1, length+3), device='cuda')
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected = model(ids, use_cache=False).logits
        with pytest.raises(RuntimeError, match='intentional'):
            with qwen3_triton_inference(model, fuse_qk=True, compact_gqa=True,
                    static_decode=True, packed_qkv=True, direct_k_cache=True):
                runner = StaticDecodeRunner(model, 1, length+3)
                runner.prefill(ids[:, :length])
                runner.capture(ids[:, length:length+1])
                for pos in range(length, length+3):
                    out = runner.decode(ids[:, pos:pos+1])
                    torch.testing.assert_close(out, expected[:, pos:pos+1], atol=.01, rtol=.02)
                raise RuntimeError('intentional')
        assert '_update_causal_mask' not in model.model.__dict__
        assert not any('_packed_qkv' in m.__dict__ or 'forward' in m.__dict__ for m in model.modules())


@pytest.mark.parametrize('tokens', [128, 512])
def test_prefill_mlp_real_width(tokens):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    from model.hf_inference import make_mlp_forward
    from types import MethodType
    c = config()
    c.hidden_size, c.intermediate_size = 2048, 6144
    mlp = Qwen3MLP(c).cuda().half().eval()
    with torch.inference_mode():
        x = torch.randn(1, tokens, 2048, device='cuda', dtype=torch.float16)
        expected = mlp(x)
        actual = MethodType(make_mlp_forward(decode_mlp=True, prefill_mlp=True), mlp)(x)
        rel = (actual.float()-expected.float()).norm()/expected.float().norm()
        assert rel < .002
