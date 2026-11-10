import pytest

from scripts.analyze_fsdp_trace import intersection, merge, summarize


def kernel(name, start, length, device=0):
    return dict(cat='kernel', ph='X', name=name, ts=start, dur=length, args={'device': device})


def test_merge_and_intersection():
    assert merge([(5, 9), (0, 3), (2, 6), (10, 10)]) == [(0, 9)]
    assert intersection([(0, 5), (8, 10)], [(2, 9)]) == 4


def test_devices_are_not_combined_and_annotations_excluded():
    events = [kernel('ncclKernel', 0, 10), kernel('gemm', 3, 4),
              kernel('gemm', 5, 4), kernel('gemm', 0, 10, device=1),
              dict(cat='gpu_user_annotation', ph='X', name='forward', ts=0, dur=100)]
    result = summarize({'traceEvents': events})['devices']
    assert result['0']['communication_union_us'] == 10
    assert result['0']['local_union_us'] == 6
    assert result['0']['communication_local_overlap_us'] == 6
    assert result['0']['communication_without_local_us'] == 4
    assert result['0']['gpu_active_union_us'] == 10
    assert result['1']['communication_local_overlap_us'] == 0


def test_missing_device_rejected():
    event = kernel('gemm', 0, 10)
    event['args'] = {}
    with pytest.raises(ValueError, match='device'):
        summarize({'traceEvents': [event]})


def test_empty_trace_rejected():
    with pytest.raises(ValueError, match='No complete'):
        summarize({'traceEvents': []})


@pytest.mark.parametrize('start,length', [(float('nan'), 1), (0, float('inf'))])
def test_nonfinite_trace_rejected(start, length):
    with pytest.raises(ValueError, match='Nonfinite'):
        summarize({'traceEvents': [kernel('gemm', start, length)]})
