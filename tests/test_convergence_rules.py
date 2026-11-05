import pytest
from training.convergence_rules import Plateau


def test_plateau_requires_minimum_checks_and_patience():
    p=Plateau(min_delta=.01,patience=3,min_checks=6)
    for x in (3.,2.,2.,2.,2.):assert not p.update(x)[1]
    assert p.update(2.)[1]


def test_improvement_resets_patience():
    p=Plateau(min_delta=.01,patience=2,min_checks=1)
    p.update(3.);p.update(3.);assert p.update(2.9)==(True,False)
    assert not p.update(2.9)[1]
    assert p.update(2.9)[1]


def test_nonfinite_is_failure_not_convergence():
    with pytest.raises(ValueError):Plateau().update(float('nan'))
