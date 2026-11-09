import pytest
import torch

from benchmarks.bf16_scratch_pack import pack_owned_scratch, packing_stages


@pytest.mark.parametrize('count', [1, 2, 3, 15, 16, 17, 127, 129, 1025, 4097])
def test_schedule_disjoint_and_complete(count):
    covered = 0
    for lo, hi in packing_stages(count):
        assert lo == covered
        assert lo < hi <= count
        assert hi - 1 < 2 * lo + 1
        covered = hi
    assert covered == count


def test_cpu_all_finite_bf16_bit_patterns_and_storage():
    # Exhaustive BF16 bit patterns include signed zero, subnormals and extrema.
    source = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    source = source[torch.isfinite(source)]
    scratch = source.float()
    address = scratch.data_ptr()
    send, receive = pack_owned_scratch(scratch, allow_mutation=True)
    assert torch.equal(send.view(torch.int16), source.view(torch.int16))
    assert send.data_ptr() == address
    assert receive.data_ptr() == address + source.numel() * 2
    assert send.numel() == receive.numel() == source.numel()
    assert send.untyped_storage().nbytes() == source.numel() * 4
    # A receive write cannot corrupt the already packed send region.
    receive.zero_()
    assert torch.equal(send.view(torch.int16), source.view(torch.int16))


@pytest.mark.parametrize('values', [[0.1], [float('nan')], [float('inf')]])
def test_reject_inexact_or_nonfinite_without_mutation(values):
    scratch = torch.tensor(values, dtype=torch.float32)
    before = scratch.view(torch.int32).clone()
    with pytest.raises(ValueError):
        pack_owned_scratch(scratch, allow_mutation=True)
    assert torch.equal(before, scratch.view(torch.int32))


def test_explicit_ownership_and_layout_required():
    scratch = torch.ones(8)
    with pytest.raises(ValueError):
        pack_owned_scratch(scratch)
    for bad in (scratch[:4], scratch.reshape(2, 4), scratch.bfloat16(),
                torch.empty(0), torch.ones(8, requires_grad=True)):
        with pytest.raises(ValueError):
            pack_owned_scratch(bad, allow_mutation=True)
    with pytest.raises(ValueError):
        pack_owned_scratch(scratch, allow_mutation=True, backend='triton')


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_invalid_count(count):
    with pytest.raises(ValueError):
        list(packing_stages(count))
