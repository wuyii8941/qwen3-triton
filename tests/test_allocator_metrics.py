import pytest
from training.allocator_metrics import COUNTERS,counter_deltas


def test_counter_deltas_and_unsupported_keys_are_explicit():
    r=counter_deltas(dict(num_device_alloc=10),dict(num_device_alloc=13))
    assert r['num_device_alloc']==3
    assert all(r[k] is None for k in COUNTERS if k!='num_device_alloc')
    assert all(v==0 for v in counter_deltas(dict.fromkeys(COUNTERS,3),dict.fromkeys(COUNTERS,3)).values())


@pytest.mark.parametrize('before,after',[(3,2),(-1,1),(0,True),(False,3),(1,1.5)])
def test_reject_resets_or_invalid_counter_types(before,after):
    with pytest.raises(ValueError):
        counter_deltas(dict(num_device_alloc=before),dict(num_device_alloc=after))
