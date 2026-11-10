import pytest
import torch

from ops.triton.attention_pipeline import (
    CompactGQAAttentionPipeline,
    ContiguousKVCache,
    sdpa_decode,
    sdpa_prefill,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_decode_pipeline_matches_sdpa() -> None:
    torch.manual_seed(31)
    b, hq, hkv, s, d = 1, 16, 8, 33, 128
    q = torch.randn(b, hq, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, hkv, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    pipeline = CompactGQAAttentionPipeline(hq, hkv, d)
    cache = ContiguousKVCache(b, hkv, s + 4, d, device=q.device, dtype=q.dtype)
    actual_prefill = pipeline.prefill(q, k, v, cache)
    expected_prefill = sdpa_prefill(q, k, v)
    torch.testing.assert_close(actual_prefill, expected_prefill, rtol=6e-2, atol=6e-2)
    assert cache.seq_len == s

    q_new = torch.randn(b, hq, 1, d, device="cuda", dtype=q.dtype)
    k_new = torch.randn(b, hkv, 1, d, device="cuda", dtype=q.dtype)
    v_new = torch.randn_like(k_new)
    actual_decode = pipeline.decode(q_new, k_new, v_new, cache)
    k_all, v_all = cache.view()
    expected_decode = sdpa_decode(q_new, k_all, v_all)
    torch.testing.assert_close(actual_decode, expected_decode, rtol=6e-2, atol=6e-2)
    assert cache.seq_len == s + 1
