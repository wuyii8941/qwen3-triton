import pytest

from scripts.summarize_fsdp_allocator import COUNTERS, audit_allocator


def example(expandable):
    config = f'backend:native,expandable_segments:{expandable}'
    return dict(memory_budget_gib=42, allocator=dict(backend='native',
        environment=dict(PYTORCH_ALLOC_CONF=config, PYTORCH_CUDA_ALLOC_CONF=None),
        settings=dict(expandable_segments=expandable, PYTORCH_CUDA_ALLOC_CONF=config,
            max_split_size=-1)), records=[dict(peak_reserved_gib=42,
                allocator_activity_max_rank=dict.fromkeys(COUNTERS, 0))])


def test_only_expandable_setting_may_vary():
    assert audit_allocator(example(False), False) == audit_allocator(example(True), True)


@pytest.mark.parametrize('fault', ['label', 'backend', 'environment', 'config', 'budget',
    'missing_counter', 'negative_counter', 'bool_counter'])
def test_reject_incomparable_evidence(fault):
    r = example(True)
    if fault == 'label': r['allocator']['settings']['expandable_segments'] = False
    elif fault == 'backend': r['allocator']['backend'] = 'cudaMallocAsync'
    elif fault == 'environment': r['allocator']['environment']['PYTORCH_CUDA_ALLOC_CONF'] = 'conflict'
    elif fault == 'config': r['allocator']['settings']['PYTORCH_CUDA_ALLOC_CONF'] = 'wrong'
    elif fault == 'budget': r['records'][0]['peak_reserved_gib'] = 43
    else:
        value = {'missing_counter': None, 'negative_counter': -1, 'bool_counter': True}[fault]
        r['records'][0]['allocator_activity_max_rank']['num_alloc_retries'] = value
    with pytest.raises(ValueError): audit_allocator(r, True)
