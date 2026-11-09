from pathlib import Path

import pytest

from scripts.audit_fsdp_scratch_bounded import audit, recompute_decisions


def cases():
    times = dict(native_k32=5., native_k34=4., bounded_head_k34=4.01, bounded_decoder_k34=3.8)
    return [dict(repeat=r, case=n, status='passed', median_step_seconds=t)
            for r in (0, 1) for n, t in times.items()]


def test_independent_decision_uses_strong_native():
    result = recompute_decisions(cases())
    assert not result['bounded_head_k34']['over_two_percent_both_orders']
    assert result['bounded_decoder_k34']['over_two_percent_both_orders']
    assert result['bounded_decoder_k34']['default_changed'] is False


def test_one_order_not_enough():
    rows = cases()
    for row in rows:
        if row['repeat'] == 1 and row['case'] == 'bounded_decoder_k34':
            row['status'] = 'oom'
    result = recompute_decisions(rows)['bounded_decoder_k34']
    assert result['ratios_vs_best_tested_native'][1] is None
    assert not result['over_two_percent_both_orders']


def test_missing_study_cannot_be_called_complete(tmp_path):
    with pytest.raises(ValueError, match='Incomplete'):
        audit(tmp_path)
