import random
import numpy as np
import pytest
import torch
import torch.distributed as dist
from benchmarks.dcp_reader_adapter import load_sorted_checkpoint
from benchmarks.verify_real_fsdp_resume import state_digest
from training.fsdp_checkpoint import save_checkpoint


def test_sorted_loader_preserves_optimizer_trajectory_rng_and_validation(tmp_path):
    if dist.is_initialized():pytest.skip('Requires an isolated CPU process group')
    dist.init_process_group('gloo',init_method=f'file://{tmp_path}/init',rank=0,world_size=1)
    try:
        torch.manual_seed(5);np.random.seed(5);random.seed(5)
        def make():
            m=torch.nn.Linear(3,2);return m,torch.optim.AdamW(m.parameters(),lr=.01)
        x=torch.arange(6,dtype=torch.float32).reshape(2,3)
        def step(model,opt):
            opt.zero_grad(set_to_none=True);loss=model(x).square().mean();loss.backward();opt.step()
            random.random();np.random.random();torch.rand(())
            return loss.item()
        model,opt=make()
        step(model,opt);step(model,opt)
        path=tmp_path/'checkpoint';metadata=dict(case='sorted-reader-cpu')
        save_checkpoint(path,model,opt,step=2,data_state=dict(next_update=2),metadata=metadata)
        expected_loss=step(model,opt);expected_state=state_digest(model,opt)
        expected_rng=(random.random(),np.random.random(),torch.rand(()).item())
        fresh,optimizer=make()
        with pytest.raises(RuntimeError,match='protocol/model mismatch'):
            load_sorted_checkpoint(path,fresh,optimizer,metadata=dict(case='wrong'))
        restored=load_sorted_checkpoint(path,fresh,optimizer,metadata=metadata)
        assert restored==dict(step=2,data_state=dict(next_update=2))
        assert step(fresh,optimizer)==expected_loss
        assert state_digest(fresh,optimizer)==expected_state
        assert (random.random(),np.random.random(),torch.rand(()).item())==expected_rng
        (path/'complete.json').unlink()
        with pytest.raises(RuntimeError,match='completion marker missing'):
            load_sorted_checkpoint(path,fresh,optimizer,metadata=metadata)
    finally:dist.destroy_process_group()
