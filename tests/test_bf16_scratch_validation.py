import pytest
import torch

from benchmarks.bf16_scratch_validation import reference_upcast_valid, validate_upcast_flag


def test_all_finite_bf16_upcasts():
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    finite = values[torch.isfinite(values)]
    assert finite.numel() == 65280
    assert reference_upcast_valid(finite.float())


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 0.1, 1.0001])
def test_invalid_values(value):
    x = torch.ones(129)
    x[-1] = value
    assert not reference_upcast_valid(x)


def test_signed_zero_and_bf16_subnormal():
    bits = torch.tensor([0, -2147483648, 65536, -2147418112], dtype=torch.int32)
    assert reference_upcast_valid(bits.view(torch.float32))


def test_cpu_not_accidentally_dispatched_to_cuda():
    with pytest.raises(ValueError, match='CUDA'):
        validate_upcast_flag(torch.ones(17))
    with pytest.raises(ValueError, match='CPU oracle'):
        reference_upcast_valid(torch.ones(17, dtype=torch.bfloat16))
