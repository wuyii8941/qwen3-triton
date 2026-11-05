"""Check the embedding contribution separately from the dense head gradient."""
import pytest
import torch

from benchmarks.check_tied_embedding import TiedEmbedding


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('n,dim', [(1, 64), (15, 128), (129, 130)])
def test_tied_transport(n, dim, dtype):
    torch.manual_seed(123)
    ids = (torch.arange(n, device='cuda') % 7).reshape(1, -1)
    weight = torch.randn(19, dim, device='cuda', dtype=dtype, requires_grad=True)
    upstream = torch.randn(1, n, dim, device='cuda', dtype=dtype)
    dense = torch.randn_like(weight)
    embedding, proxy = TiedEmbedding.apply(weight, ids)
    actual, = torch.autograd.grad((embedding, proxy), (weight,), (upstream, dense))
    contribution = torch.zeros_like(weight, dtype=torch.float32)
    contribution.index_add_(0, ids.flatten(), upstream.reshape(n, dim).float())
    expected = dense + contribution.to(dtype)
    tolerance = 2e-5 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    # Independently gate new gradient: dense background must not hide errors.
    embedding, proxy = TiedEmbedding.apply(weight, ids)
    isolated, = torch.autograd.grad((embedding, proxy), (weight,), (upstream, torch.zeros_like(dense)))
    torch.testing.assert_close(isolated, contribution.to(dtype), rtol=tolerance, atol=tolerance)
    untouched = torch.arange(19, device='cuda') >= min(n, 7)
    assert torch.equal(actual[untouched], dense[untouched])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('all_same', [False,True])
def test_noncontiguous_and_hot_token(all_same):
    torch.manual_seed(18)
    n,dim=129,130
    ids=torch.randint(0,19,(n,2),device='cuda').T
    if all_same: ids.fill_(3)
    weight=torch.randn(19,dim,device='cuda',dtype=torch.float32,requires_grad=True)
    dense=torch.randn(dim,19,device='cuda').T
    dy=torch.randn(dim,2*n,device='cuda').T.reshape(2,n,dim)
    output,proxy=TiedEmbedding.apply(weight,ids)
    actual,=torch.autograd.grad((output,proxy),(weight,),(dy,dense))
    # FP64 oracle separates summation rounding from indexing correctness.
    contribution=torch.zeros_like(weight,dtype=torch.float64)
    contribution.index_add_(0,ids.flatten(),dy.reshape(2*n,dim).double())
    expected=dense.double()+contribution
    torch.testing.assert_close(actual.double(),expected,rtol=2e-5,atol=1e-5)
