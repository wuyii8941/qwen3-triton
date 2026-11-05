"""Custom-op mutation schemas and fullgraph static-cache correctness."""
import pytest
import torch
from ops import compile_ops as ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_custom_op_schemas(dtype):
    with torch.inference_mode():
        x = torch.randn(2, 128, device='cuda', dtype=dtype)
        w = torch.randn(128, device='cuda', dtype=dtype)
        gate = torch.randn(64, 128, device='cuda', dtype=dtype) * .02
        cases = [
            (ops.norm, (x, w, 1e-6)),
            (ops.activation, (x, x.clone())),
            (ops.add_norm, (x, x.clone(), w, 1e-6)),
            (ops.decode_gate_up, (x, gate, gate.clone(), 1, 4)),
            (ops.gate_up, (x, gate, gate.clone(), 32, 64, 32, 4, 3, 0, True)),
        ]
        q = torch.randn(2, 3, 4, 64, device='cuda', dtype=dtype)
        k = torch.randn(2, 3, 2, 64, device='cuda', dtype=dtype)
        v = torch.randn_like(k).transpose(1, 2)
        co = torch.randn(1, 3, 64, device='cuda', dtype=dtype)
        si = torch.randn_like(co)
        nw = torch.ones(64, device='cuda', dtype=dtype)
        kc = torch.zeros(2, 8, 2, 64, device='cuda', dtype=dtype)
        vc = torch.zeros_like(kc)
        positions = torch.arange(3, device='cuda')
        args = (q, k, nw, nw.clone(), co, si, 1e-6)
        cases.extend([
            (ops.qk_rope, args),
            (ops.qk_rope_store_k, (*args, kc, positions)),
            (ops.qk_rope_store_kv, (*args, kc, positions, v, vc)),
        ])
        for op, inputs in cases:
            # AOT dispatch is separately checked by the full Inductor test.
            torch.library.opcheck(op, inputs,
                test_utils=('test_schema', 'test_faketensor'))


@pytest.mark.parametrize('variant', ['torch', 'triton'])
def test_compiled_incremental_graph(variant):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from torch._dynamo.backends.registry import lookup_backend
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from model.hf_inference import qwen3_triton_inference, pack_qkv_weights
    from model.torch_static_reference import qwen3_torch_static_reference
    from model.static_decode import StaticDecodeRunner

    torch.manual_seed(151)
    c = Qwen3Config(vocab_size=256, hidden_size=256, intermediate_size=512,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=64)
    c._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(c).cuda().half().eval()
    ids = torch.randint(256, (2, 20), device='cuda')
    count = []
    def backend(gm, inputs):
        count.append(len(list(gm.graph.nodes)))
        return lookup_backend('inductor')(gm, inputs, config_patches={
            'triton.cudagraphs': False, 'compile_threads': 4, 'emulate_precision_casts': True})

    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected = model(ids, use_cache=False).logits
        context = (qwen3_torch_static_reference(model) if variant == 'torch' else
            qwen3_triton_inference(model, fuse_qk=True, compact_gqa=True, static_decode=True,
                packed_qkv=True, packed_weights=pack_qkv_weights(model),
                direct_k_cache=True, direct_v_cache=True, fuse_residual=True))
        with context:
            runner = StaticDecodeRunner(model, 2, 24)
            runner._forward = torch.compile(runner._forward, backend=backend, fullgraph=True, dynamic=False)
            runner.prefill(ids[:, :15])
            runner.capture(ids[:, 15:16])
            # Refill same-size prompt twice: no context rebinding or recompiles.
            runner.prefill(ids[:, :15])
            compiled = len(count)
            for _ in range(2):
                first = runner.prefill(ids[:, :15])
                torch.testing.assert_close(first, expected[:, 14:15], atol=.02, rtol=.04)
                for pos in range(15, 20):
                    out = runner.decode(ids[:, pos:pos+1])
                    torch.testing.assert_close(out, expected[:, pos:pos+1], atol=.02, rtol=.04)
            assert compiled >= 2, 'Both prefill and decode must be compiled'
            assert len(count) == compiled, 'Timing would be invalidated by recompiles'
        assert not any('forward' in m.__dict__ for m in model.modules())


def test_hybrid_shared_cache():
    import copy
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from model.hybrid_decode import HybridDecodeRunner
    from model.hf_inference import qwen3_triton_inference
    from model.torch_static_reference import qwen3_torch_static_reference
    from torch.nn.attention import sdpa_kernel, SDPBackend
    c=Qwen3Config(vocab_size=256,hidden_size=256,intermediate_size=512,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=64)
    c._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(c).cuda().half().eval()
    twin=copy.deepcopy(model,{id(p):p for p in model.parameters()})
    with pytest.raises(ValueError,match='share'):
        HybridDecodeRunner(model,copy.deepcopy(model),2,32)
    ids=torch.randint(256,(2,20),device='cuda')
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ref=model(ids,use_cache=False).logits
        with qwen3_torch_static_reference(model), qwen3_triton_inference(twin,
                fuse_qk=True,compact_gqa=True,static_decode=True,direct_k_cache=True,
                direct_v_cache=True,fuse_residual=True):
            r=HybridDecodeRunner(model,twin,2,32)
            r.compile_prefill(backend='inductor')
            r.prefill(ids[:,:15]);r.capture(ids[:,15:16])
            assert r.prefill_runner.cache is r.decode_runner.cache
            for _ in range(2):
                r.prefill(ids[:,:15])
                for pos in range(15,20):
                    torch.testing.assert_close(r.decode(ids[:,pos:pos+1]),ref[:,pos:pos+1],atol=.02,rtol=.04)
