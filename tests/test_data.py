import numpy as np
import pytest
import torch

from training.data import Blocks


def test_blocks_match_historical_permutation_and_drop_tail(tmp_path):
    path = tmp_path / 'corpus.bin'
    np.arange(51, dtype='<u4').tofile(path)
    evaluation = Blocks(path, 5)
    training = Blocks(path, 5, training=True, seed=42)
    assert len(evaluation) == 10
    for i in range(10):
        torch.testing.assert_close(evaluation[i], torch.arange(i * 5, i * 5 + 5))
        index = (i * training.stride + 42) % 10
        torch.testing.assert_close(training[i], evaluation[index])
    assert len({tuple(training[i].tolist()) for i in range(10)}) == 10


def test_blocks_reject_bad_sequence_length(tmp_path):
    for seq in (0, 1, -1, 2.5):
        with pytest.raises(ValueError):
            Blocks(tmp_path / 'missing.bin', seq)
