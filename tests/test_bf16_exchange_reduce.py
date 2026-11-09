import pytest
import torch
from benchmarks.bf16_exchange_reduce import sum_received


@pytest.mark.parametrize('average',[True,False])
def test_fp32_accumulation_and_output_dtype(average):
    x=torch.tensor([1.,-2.,3.,4.,5.,6.,-7.,8.],dtype=torch.bfloat16)
    result=sum_received(x,4,average=average)
    expected=(x.reshape(4,2).float()*(.25 if average else 1.)).sum(0)
    torch.testing.assert_close(result,expected,atol=0,rtol=0)
    assert result.dtype==torch.float32


def test_scale_before_sum_avoids_unnecessary_average_overflow():
    x=torch.full((8,),torch.finfo(torch.bfloat16).max,dtype=torch.bfloat16)
    assert torch.isfinite(sum_received(x,4)).all()


@pytest.mark.parametrize('x,world',[(torch.ones(8),4),(torch.ones(7,dtype=torch.bfloat16),4),
    (torch.ones(8,dtype=torch.bfloat16),1),(torch.ones(4,4,dtype=torch.bfloat16).T,4)])
def test_reject_fp32_cast_and_unsupported_layout(x,world):
    with pytest.raises(ValueError):sum_received(x,world)
