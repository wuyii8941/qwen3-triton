import pytest
import torch

from benchmarks.bf16_scratch_bounded import (
    BoundedScratchExchangeReduceScatter, pack_owned_scratch_bounded,
)


def test_ownership_permission_is_not_implicit():
    with pytest.raises(ValueError, match='permission'):
        pack_owned_scratch_bounded(torch.ones(8))


def test_cpu_is_not_implicitly_moved_to_gpu():
    with pytest.raises(ValueError, match='CUDA'):
        pack_owned_scratch_bounded(torch.ones(8), allow_mutation=True)


@pytest.mark.parametrize('options', [dict(original_bf16=False),
    dict(original_bf16=True, accumulation=2), dict(original_bf16=True, debug=False)])
def test_adapter_keeps_restrictions(options):
    with pytest.raises(ValueError):
        BoundedScratchExchangeReduceScatter(**options)
