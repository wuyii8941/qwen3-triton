"""Preserve official per-tensor quantization/state and missing-gradient semantics."""
import copy
import pytest
import torch

pytest.importorskip('flashoptim')
from flashoptim import FlashAdamW
from training.multi_tensor_flashoptim import MultiTensorFlashAdamW


def compare(a, b, oa, ob):
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        sa, sb = oa.state[x], ob.state[y]
        assert sa.keys() == sb.keys()
        for key in sa:
            if hasattr(sa[key], 'kernel_tensor'):
                torch.testing.assert_close(sa[key].kernel_tensor, sb[key].kernel_tensor, rtol=0, atol=0)
                torch.testing.assert_close(sa[key].kernel_scales_or_self, sb[key].kernel_scales_or_self, rtol=0, atol=0)
            else:
                torch.testing.assert_close(sa[key], sb[key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('master_bits', [None, 24])
@pytest.mark.parametrize('quantize', [True, False])
@pytest.mark.parametrize('batch_all', [False,True,'vectorized'])
def test_matches_official(master_bits, quantize, batch_all):
    torch.manual_seed(19)
    sizes = [1, 31, 32, 128, 131, 1024, 2048, 32769, 70000]
    a = [torch.nn.Parameter(torch.randn(n, device='cuda', dtype=torch.bfloat16)*.02) for n in sizes]
    b = [torch.nn.Parameter(x.detach().clone()) for x in a]
    options = dict(lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=.01,
                   decouple_lr=False, master_weight_bits=master_bits, quantize=quantize)
    batch_options=dict(max_numel=2**31-1,persistent=True,vectorized=batch_all=='vectorized') if batch_all else {}
    oa, ob = FlashAdamW(a, **options), MultiTensorFlashAdamW(b, **batch_options, **options)
    for step in range(12):
        for i, (x, y) in enumerate(zip(a, b)):
            g = torch.randn_like(x)*.001
            x.grad = g if (step+i) % 4 else None
            y.grad = g.clone() if x.grad is not None else None
        if step == 5:
            oa.param_groups[0]['lr'] = ob.param_groups[0]['lr'] = 3e-6
        oa.step()
        ob.step()
        compare(a, b, oa, ob)
        if step == 6:
            # Official checkpoint format; fresh subclass must rebuild metadata.
            saved = copy.deepcopy(oa.state_dict())
            ob = MultiTensorFlashAdamW(b, **batch_options, **options)
            ob.load_state_dict(saved)
            compare(a, b, oa, ob)
        if step == 8:
            # Reverse direction is equally important for rollback.
            oa = FlashAdamW(a, **options)
            oa.load_state_dict(copy.deepcopy(ob.state_dict()))
            compare(a, b, oa, ob)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_parameter_groups_and_dtype_fallback():
    torch.manual_seed(91)
    a=[torch.nn.Parameter(torch.randn(2048,device='cuda',dtype=d))
       for d in (torch.bfloat16,torch.float16,torch.float32,torch.bfloat16)]
    b=[torch.nn.Parameter(p.detach().clone()) for p in a]
    def groups(params):
        return [dict(params=params[:2],lr=1e-5,betas=(.9,.999)),
                dict(params=params[2:],lr=3e-5,betas=(.8,.95),weight_decay=.02)]
    opts=dict(master_weight_bits=None,decouple_lr=True)
    oa,ob=FlashAdamW(groups(a),**opts),MultiTensorFlashAdamW(groups(b),**opts)
    for step in range(4):
        for x,y in zip(a,b):
            x.grad=torch.randn_like(x)*.01;y.grad=x.grad.clone()
        if step==2:
            for opt in (oa,ob):
                for group in opt.param_groups: group['lr']*=.5
        oa.step();ob.step();compare(a,b,oa,ob)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_persistent_grid_stride_large_tail():
    torch.manual_seed(92)
    a=[torch.nn.Parameter(torch.randn(n,device='cuda',dtype=torch.bfloat16)*.02)
       for n in (128,2048,2048*2048+1)]
    b=[torch.nn.Parameter(p.detach().clone()) for p in a]
    opts=dict(lr=1e-5,master_weight_bits=None,decouple_lr=False)
    oa=FlashAdamW(a,**opts)
    ob=MultiTensorFlashAdamW(b,max_numel=2**31-1,persistent=True,vectorized=True,**opts)
    for _ in range(5):
        for x,y in zip(a,b):
            x.grad=torch.randn_like(x)*.001;y.grad=x.grad.clone()
        oa.step();ob.step();compare(a,b,oa,ob)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_async_metadata_without_per_step_synchronization():
    torch.manual_seed(93)
    a=[torch.nn.Parameter(torch.randn(n,device='cuda',dtype=torch.bfloat16)*.02)
       for n in (128,2048,1048577)]
    b=[torch.nn.Parameter(p.detach().clone()) for p in a]
    opts=dict(lr=1e-5,master_weight_bits=None,decouple_lr=False)
    oa=FlashAdamW(a,**opts)
    ob=MultiTensorFlashAdamW(b,max_numel=2**31-1,persistent=True,vectorized=True,async_metadata=True,**opts)
    for step in range(64):
        for i,(x,y) in enumerate(zip(a,b)):
            grad=torch.randn_like(x)*.001
            x.grad=grad if (i+step)%5 else None
            y.grad=grad if x.grad is not None else None
        oa.step();ob.step()
    # No compare/synchronize/.cpu during the loop: exercise DMA lifetime and
    # per-parameter counters while the CPU queues ahead of the GPU.
    torch.cuda.synchronize();compare(a,b,oa,ob)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_unaligned_contiguous_storage_falls_back():
    torch.manual_seed(94)
    a=[torch.nn.Parameter(torch.randn(132,device='cuda',dtype=torch.bfloat16)[1:])]
    b=[torch.nn.Parameter(torch.empty(132,device='cuda',dtype=torch.bfloat16)[1:])]
    with torch.no_grad(): b[0].copy_(a[0])
    assert a[0].is_contiguous() and a[0].data_ptr()%16!=0
    opts=dict(lr=1e-5,master_weight_bits=None,decouple_lr=False)
    oa=FlashAdamW(a,**opts)
    ob=MultiTensorFlashAdamW(b,max_numel=2**31-1,persistent=True,vectorized=True,async_metadata=True,**opts)
    for _ in range(4):
        g=torch.randn(132,device='cuda',dtype=torch.bfloat16)[1:]
        a[0].grad=g;b[0].grad=g
        oa.step();ob.step();compare(a,b,oa,ob)
    assert not ob._metadata_buffers
