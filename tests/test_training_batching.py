import numpy as np
import pytest
from torch.utils.data import DataLoader
from training.batching import accumulation_groups
from training.flashoptim_convergence_job import Blocks


def test_groups_drop_incomplete_tail():
    assert list(accumulation_groups(range(5),2))==[(0,1),(2,3)]
    with pytest.raises(ValueError):list(accumulation_groups([],0))


def test_resume_offset_tracks_samples_not_optimizer_steps(tmp_path):
    path=tmp_path/'train.bin';path.write_bytes(np.arange(40,dtype='<u4').tobytes())
    dataset=Blocks(path,4,training=True,seed=7)
    groups=list(accumulation_groups(DataLoader(dataset,batch_size=2,drop_last=True),2))
    resumed=Blocks(path,4,training=True,seed=7,offset=4)
    next_group=next(accumulation_groups(DataLoader(resumed,batch_size=2,drop_last=True),2))
    for a,b in zip(groups[1],next_group):assert (a==b).all()
