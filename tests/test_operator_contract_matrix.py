"""Core low-precision contracts against FP32 math, not HF cast-boundary parity."""
import pytest
import torch

from ops.reference.rms_norm import rms_norm_reference
from ops.reference.rope import apply_rope_reference
from ops.reference.swiglu import swiglu_reference
from ops.triton.rms_norm import rms_norm_triton
from ops.triton.rope import apply_rope_triton
from ops.triton.silu_and_mul import silu_and_mul_triton


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA contract matrix')
SHAPES = [(1, 1), (2, 17), (1, 129)]


def check(actual_fn, oracle_fn, inputs, dtype):
    snapshots = [value.detach().clone() for value in inputs]
    oracle_inputs = [value.detach().float().requires_grad_() for value in inputs]
    actual = actual_fn(*inputs)
    expected = oracle_fn(*oracle_inputs)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    upstream = tuple(torch.randn_like(value) for value in actual)
    # Oracle receives the same quantized upstream gradients. All internal
    # operations stay FP32; the comparison rounds only the final output/grad.
    expected_grads = torch.autograd.grad(expected, oracle_inputs, tuple(value.float() for value in upstream))
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    tolerance = 3e-3 if dtype == torch.float16 else 3e-2
    for observed, reference in zip(actual + actual_grads, expected + expected_grads):
        assert observed.dtype == dtype
        assert observed.shape == reference.shape
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference.to(dtype), rtol=tolerance, atol=tolerance)
    for observed, snapshot in zip(inputs, snapshots):
        torch.testing.assert_close(observed.detach(), snapshot, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('batch,seq', SHAPES)
@pytest.mark.parametrize('width', [128, 2048])
def test_rmsnorm_matrix(batch, seq, width, dtype):
    torch.manual_seed(20261109)
    x = torch.randn(batch, seq, width, device='cuda', dtype=dtype, requires_grad=True)
    weight = torch.randn(width, device='cuda', dtype=dtype, requires_grad=True)
    check(lambda x, w: rms_norm_triton(x, w, 1e-6),
          lambda x, w: rms_norm_reference(x, w, 1e-6), (x, weight), dtype)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('batch,seq', SHAPES)
@pytest.mark.parametrize('width', [64, 128])
def test_rope_gqa_matrix(batch, seq, width, dtype):
    torch.manual_seed(20261110)
    q = torch.randn(batch, 4, seq, width, device='cuda', dtype=dtype, requires_grad=True)
    k = torch.randn(batch, 2, seq, width, device='cuda', dtype=dtype, requires_grad=True)
    # Independent half-coordinates deliberately test the general transpose,
    # rather than letting duplicated sine values hide a backward indexing bug.
    cos = torch.randn(batch, seq, width, device='cuda', dtype=dtype)
    sin = torch.randn_like(cos)
    check(lambda q, k: apply_rope_triton(q, k, cos, sin),
          lambda q, k: apply_rope_reference(q, k, cos.float(), sin.float()), (q, k), dtype)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('batch,seq', SHAPES)
@pytest.mark.parametrize('width', [257, 6144])
def test_swiglu_matrix(batch, seq, width, dtype):
    torch.manual_seed(20261111)
    gate = torch.randn(batch, seq, width, device='cuda', dtype=dtype, requires_grad=True)
    up = torch.randn_like(gate, requires_grad=True)
    check(silu_and_mul_triton, swiglu_reference, (gate, up), dtype)
