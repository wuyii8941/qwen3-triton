import pytest

from scripts.screen_fsdp_retention_pressure import CASES, command, decisions, helpers


@pytest.mark.parametrize('name', CASES)
def test_matched_unprofiled_contract(name):
    cmd = command(name, 'unused')
    assert '--profile' not in cmd and '--reference-check' in cmd
    assert cmd[cmd.index('--memory-budget-gib') + 1] == '41.875'
    assert cmd[cmd.index('--keep-full-layers') + 1] == str(CASES[name][0])
    assert ('--exchange-reduce' in cmd) == CASES[name][1]


def records(speed=5, retries=1, syncs=1):
    return [dict(repeat=repeat, case=name, median_step_seconds=speed if name.endswith('24') else 6,
        counter_medians=dict(num_alloc_retries=retries if name.endswith('24') else 3,
                             num_sync_all_streams=syncs if name.endswith('24') else 3))
        for repeat in (0, 1) for name in CASES]


def test_speed_and_pressure_hypothesis_are_distinct():
    assert decisions(records())['decoder_k24']['worth_further_validation']
    assert decisions(records())['decoder_k24']['supports_pressure_hypothesis']
    assert not decisions(records(speed=6.1))['decoder_k24']['worth_further_validation']
    assert decisions(records(retries=3))['decoder_k24']['worth_further_validation']
    assert not decisions(records(retries=3))['decoder_k24']['supports_pressure_hypothesis']
    assert not decisions(records(syncs=3))['decoder_k24']['supports_pressure_hypothesis']


def test_one_positive_order_is_not_enough():
    rows = records()
    rows[-1]['median_step_seconds'] = 7
    assert not decisions(rows)['decoder_k24']['worth_further_validation']


def test_guard_base_worker_and_screen():
    assert 'benchmarks/compare_fsdp_training.py' in helpers()
    assert 'scripts/screen_fsdp_retention_pressure.py' in helpers()
