import pytest
import torch

from ops.reference.swiglu import swiglu_reference
from model.config import TinyQwenConfig
from model.qwen_tiny import QwenMLP


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_swiglu_reference_matches_eager(dtype: torch.dtype) -> None:
    torch.manual_seed(40)
    gate = torch.randn(2, 3, 257, dtype=dtype, requires_grad=True)
    up = torch.randn_like(gate, requires_grad=True)
    actual = swiglu_reference(gate, up)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(actual, expected)


def test_swiglu_reference_backward() -> None:
    torch.manual_seed(41)
    gate = torch.randn(4, 129, requires_grad=True)
    up = torch.randn_like(gate, requires_grad=True)
    grad = torch.randn_like(gate)
    actual = swiglu_reference(gate, up)
    actual.backward(grad)
    gate_expected = gate.detach().requires_grad_()
    up_expected = up.detach().requires_grad_()
    expected = torch.nn.functional.silu(gate_expected) * up_expected
    expected.backward(grad)
    torch.testing.assert_close(gate.grad, gate_expected.grad)
    torch.testing.assert_close(up.grad, up_expected.grad)


def test_swiglu_reference_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        swiglu_reference(torch.ones(2, 3), torch.ones(2, 4))


def test_packed_gate_up_matches_two_projections() -> None:
    torch.manual_seed(42)
    cfg = TinyQwenConfig(hidden_size=16, intermediate_size=32, packed_gate_up=False)
    packed_cfg = TinyQwenConfig(hidden_size=16, intermediate_size=32, packed_gate_up=True)
    separate = QwenMLP(cfg)
    packed = QwenMLP(packed_cfg)
    with torch.no_grad():
        packed.gate_up_proj.weight[:32].copy_(separate.gate_proj.weight)
        packed.gate_up_proj.weight[32:].copy_(separate.up_proj.weight)
        packed.down_proj.weight.copy_(separate.down_proj.weight)
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(packed(x), separate(x), rtol=1e-6, atol=1e-6)
