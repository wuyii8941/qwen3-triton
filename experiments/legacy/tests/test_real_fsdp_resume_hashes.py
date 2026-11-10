import torch
from benchmarks.verify_real_fsdp_resume import tensor_digest,state_digest,resource_plan


def test_hash_checks_bits_dtype_shape_and_noncontiguous_values():
    x=torch.arange(12,dtype=torch.float32).reshape(3,4).T
    assert tensor_digest(x)==tensor_digest(x.contiguous())
    assert tensor_digest(x)!=tensor_digest(x.reshape(-1))
    assert tensor_digest(x)!=tensor_digest(x.double())
    y=x.clone();y[0,0]=-0.
    assert torch.equal(x,y) and tensor_digest(x)!=tensor_digest(y)
    assert tensor_digest(torch.tensor(1.))['local_shape']==[]
    tensor_digest(torch.ones(2,dtype=torch.bfloat16))


def test_state_hash_covers_moments_and_optimizer_settings():
    model=torch.nn.Linear(2,2);opt=torch.optim.AdamW(model.parameters(),lr=.01)
    model(torch.ones(1,2)).sum().backward();opt.step()
    before=state_digest(model,opt)
    assert before==state_digest(model,opt)
    opt.state[model.weight]['exp_avg'][0,0]+=1
    assert before!=state_digest(model,opt)
    before=state_digest(model,opt);opt.param_groups[0]['lr']=.02
    assert before!=state_digest(model,opt)
    model.register_buffer('test_buffer',torch.zeros(2),persistent=False)
    before=state_digest(model,opt);model.test_buffer[0]=1
    assert before!=state_digest(model,opt)


def test_resource_plan_refuses_low_disk_before_checkpoint(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import benchmarks.verify_real_fsdp_resume as module
    monkeypatch.setattr(module.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    plan=resource_plan(torch.nn.Linear(2,2),tmp_path)
    assert not plan['resources_sufficient'] and plan['disk_required_bytes']>0
