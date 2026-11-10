import pytest
import torch
from training.triton_adamw import TritonAdamW

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('decay',[0.,.01])
def test_native_fused_updates_and_states(dtype,decay):
    torch.manual_seed(71)
    originals=[torch.randn(n,device='cuda',dtype=dtype) for n in (1,129,16387)]
    p=[torch.nn.Parameter(x.clone()) for x in originals]
    q=[torch.nn.Parameter(x.clone()) for x in originals]
    native=torch.optim.AdamW(p,lr=1e-3,weight_decay=decay,fused=True)
    candidate=TritonAdamW(q,lr=1e-3,weight_decay=decay)
    for step in range(20):
        for i,(x,y) in enumerate(zip(p,q)):
            if i==0 and step%3==0:x.grad=y.grad=None
            else:
                grad=torch.randn_like(x)*.1
                x.grad=grad;y.grad=grad.clone()
        native.step();candidate.step()
        for x,y in zip(p,q):
            tol=2e-6 if dtype==torch.float32 else (.004 if dtype==torch.bfloat16 else .0005)
            torch.testing.assert_close(y,x,atol=tol,rtol=tol)
            if not native.state[x]:continue
            assert candidate.state[y]['step']==int(native.state[x]['step'])
            for key in ('exp_avg','exp_avg_sq'):
                a,b=candidate.state[y][key].float(),native.state[x][key].float()
                rel=(a-b).norm()/b.norm().clamp_min(1e-12)
                assert rel < (.01 if dtype==torch.bfloat16 else .002), (dtype,key,float(rel))


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_actual_update_error_not_just_parameter_error(dtype):
    torch.manual_seed(72)
    init=(torch.randn(32771,device='cuda')*.01).to(dtype)
    p=torch.nn.Parameter(init.clone());q=torch.nn.Parameter(init.clone())
    a=torch.optim.AdamW([p],lr=1e-5,fused=True)
    b=TritonAdamW([q],lr=1e-5,math_mode='rounded')
    for _ in range(20):
        oldp=p.detach().float().clone();oldq=q.detach().float().clone()
        grad=torch.randn_like(p)*.01;p.grad=grad;q.grad=grad.clone()
        a.step();b.step()
        da=p.detach().float()-oldp;db=q.detach().float()-oldq
        rel=(db-da).norm()/da.norm().clamp_min(1e-20)
        assert rel < (.03 if dtype==torch.bfloat16 else .001),float(rel)


def test_checkpoint_resume():
    import copy
    p=torch.nn.Parameter(torch.ones(129,device='cuda'))
    a=TritonAdamW([p]);p.grad=torch.full_like(p,.01);a.step()
    q=torch.nn.Parameter(p.detach().clone());b=TritonAdamW([q])
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    for _ in range(4):
        p.grad=torch.full_like(p,.02);q.grad=p.grad.clone();a.step();b.step()
    torch.testing.assert_close(p,q,atol=0,rtol=0)


def test_mode_survives_checkpoint():
    import copy
    p=torch.nn.Parameter(torch.ones(17,device='cuda'))
    a=TritonAdamW([p],math_mode='legacy');p.grad=torch.full_like(p,.01);a.step()
    q=torch.nn.Parameter(p.detach().clone());b=TritonAdamW([q],math_mode='rounded')
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    assert b.param_groups[0]['math_mode']=='legacy'
    p.grad=torch.full_like(p,.02);q.grad=p.grad.clone();a.step();b.step()
    torch.testing.assert_close(p,q,atol=0,rtol=0)


@pytest.mark.parametrize('kwargs',[{'lr':float('nan')},{'eps':-1.},{'betas':(.9,1.)}])
def test_reject_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):TritonAdamW([torch.nn.Parameter(torch.ones(1,device='cuda'))],**kwargs)


def test_reject_unsupported_group():
    with pytest.raises(ValueError,match='Unsupported'):
        TritonAdamW([{'params':[torch.nn.Parameter(torch.ones(1,device='cuda'))],'amsgrad':True}])
