import numpy as np
import pytest
import torch
from training.convergence_job import Blocks,durable_save


def test_resume_data_order_is_same(tmp_path):
    path=tmp_path/'tokens.bin'
    np.arange(512,dtype='<u4').tofile(path)
    original=Blocks(path,16,training=True)
    resumed=Blocks(path,16,offset=13,training=True)
    for i in range(19):torch.testing.assert_close(original[(i+13)%32],resumed[i])
    assert len({int(original[i][0]) for i in range(32)})==32


def test_eval_is_ordered_and_separate(tmp_path):
    path=tmp_path/'tokens.bin';np.arange(37,dtype='<u4').tofile(path)
    data=Blocks(path,16)
    assert len(data)==2
    torch.testing.assert_close(data[1],torch.arange(16,32))


def test_atomic_checkpoint_roundtrip(tmp_path):
    path=tmp_path/'last.pt'
    durable_save({'step':1,'x':torch.arange(7)},path)
    durable_save({'step':2,'x':torch.arange(9)},path)
    saved=torch.load(path,weights_only=True)
    assert saved['step']==2 and len(saved['x'])==9
    assert not path.with_suffix('.tmp').exists()


def test_too_short_corpus_rejected(tmp_path):
    path=tmp_path/'tokens.bin';np.arange(3,dtype='<u4').tofile(path)
    with pytest.raises(ValueError):Blocks(path,16)


def test_training_seed_changes_order_but_not_eval(tmp_path):
    path=tmp_path/'tokens.bin';np.arange(512,dtype='<u4').tofile(path)
    first=Blocks(path,16,training=True,seed=20261102)
    second=Blocks(path,16,training=True,seed=20261103)
    assert not torch.equal(first[0],second[0])
    resumed=Blocks(path,16,offset=7,training=True,seed=20261103)
    torch.testing.assert_close(second[7],resumed[0])
    torch.testing.assert_close(Blocks(path,16,seed=1)[0],Blocks(path,16,seed=2)[0])
