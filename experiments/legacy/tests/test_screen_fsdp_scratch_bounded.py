from pathlib import Path

import pytest

from scripts.screen_fsdp_scratch_bounded import CASES, command, decisions


@pytest.mark.parametrize('name', list(CASES))
def test_fixed_real_workload_and_reference(name):
    cmd = command(CASES[name], Path('example'))
    assert '--tiny' not in cmd and '--reference-check' in cmd
    for flag, value in (('--memory-budget-gib', '40'), ('--steps', '12'),
                        ('--warmup', '3'), ('--seq', '512'), ('--batch', '1')):
        assert cmd[cmd.index(flag) + 1] == value
    scope = CASES[name][1]
    assert ('--bounded-exchange-scope' in cmd) == bool(scope)
    if scope:
        assert cmd[-1] == scope


def rows():
    times = dict(native_k32=5., native_k34=4., bounded_head_k34=4.01, bounded_decoder_k34=3.8)
    return [dict(repeat=r, case=n, status='passed', median_step_seconds=times[n])
            for r in (0, 1) for n in CASES]


def test_strongest_native_not_weakest_control():
    result = decisions(rows())
    assert not result['bounded_head_k34']['over_two_percent_both_orders']
    assert result['bounded_decoder_k34']['over_two_percent_both_orders']
    assert not result['bounded_decoder_k34']['default_changed']
    assert not result['bounded_decoder_k34']['convergence_proven']


def test_both_orders_required_and_oom_not_win():
    records = rows()
    for row in records:
        if row['repeat'] == 1 and row['case'] == 'bounded_decoder_k34':
            row['status'] = 'oom'
    result = decisions(records)['bounded_decoder_k34']
    assert result['ratios_vs_best_tested_native'][1] is None
    assert not result['over_two_percent_both_orders']
