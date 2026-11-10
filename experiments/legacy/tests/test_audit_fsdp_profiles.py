import json

import pytest

from scripts.audit_fsdp_profiles import audit, step_summary


def test_step_summary_uses_only_measured_deltas():
    rows = [dict(warmup=True, peak_allocated_gib=99, peak_reserved_gib=100,
                 allocator_activity_max_rank=dict(retries=99, missing=None))]
    for value in (3, 4, 5):
        rows.append(dict(warmup=False, peak_allocated_gib=value, peak_reserved_gib=value + 1,
                         allocator_activity_max_rank=dict(retries=value, missing=None)))
    result = step_summary(dict(records=rows))
    assert result['counter_medians'] == dict(retries=4, missing=None)
    assert result['peak_allocated_gib'] == 5
    assert result['peak_reserved_gib'] == 6


@pytest.mark.parametrize('complete,order', [(False, []), (True, ['native_a']),
    (True, ['decoder_a', 'native_a', 'decoder_b', 'native_b'])])
def test_partial_or_reordered_campaign_rejected(tmp_path, complete, order):
    (tmp_path / 'result.json').write_text(json.dumps(dict(complete=complete,
        records=[dict(case=name) for name in order], defaults_changed=False, checkpoints_written=False)))
    with pytest.raises(ValueError, match='Complete ABBA'):
        audit(tmp_path)
