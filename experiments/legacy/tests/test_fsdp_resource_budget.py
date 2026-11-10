import pytest
from scripts.fsdp_resource_budget import budget_ready


def test_allocator_budget_plus_explicit_margin():
    assert budget_ready([(3000,45568,0)]*4)
    assert not budget_ready([(3000,45567,0)]*4)
    assert not budget_ready([(3000,45568,10)]*4)
    assert not budget_ready([(3000,45568,0)]*3)


@pytest.mark.parametrize('budget,margin',[(0,2.5),(42,0),(42,1.9),(42,float('nan')),(float('inf'),2.5)])
def test_invalid_or_insufficient_safety_margin_rejected(budget,margin):
    with pytest.raises(ValueError):budget_ready([(0,48000,0)]*4,
        allocator_gib=budget,free_margin_gib=margin)
