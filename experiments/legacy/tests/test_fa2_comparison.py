import torch
from benchmarks.benchmark_fa2_comparison import error


def test_error_metric_checks_nonfinite():
    assert not error(torch.tensor([float('nan')]),torch.ones(1))['finite']


def test_error_metric_does_not_confuse_absolute_and_relative_error():
    result=error(torch.tensor([3.,4.]),torch.tensor([0.,4.]))
    assert result['max_abs']==3.
    assert result['relative_l2']==.75
