import pytest

from benchmarks.fsdp_scratch_bounded_quality import configure


def test_native_quality_retains_explicit_policy_and_protocol():
    args, base = configure(['--backend', 'official', '--output', 'example',
                            '--memory-budget-gib', '40', '--keep-full-layers', '34',
                            '--checkpoint-every', '1', '--protocol', 'protocol.json'])
    assert args.bounded_exchange_scope is None and args.memory_budget_gib == 40
    assert '--memory-budget-gib' not in base  # wrapper owns cap, base has no such CLI
    assert base[base.index('--keep-full-layers') + 1] == '34'
    assert base[-1] == 'protocol.json'


def test_bounded_tiny_preflight_is_not_convergence():
    args, base = configure(['--backend', 'official', '--output', 'example', '--memory-budget-gib', '40',
                            '--bounded-exchange-scope', 'decoder', '--tiny', '--preflight-steps', '2'])
    assert args.bounded_exchange_scope == 'decoder'
    assert '--tiny' in base and base[base.index('--preflight-steps') + 1] == '2'
    assert '--bounded-exchange-scope' not in base


def test_original_exchange_flags_rejected():
    with pytest.raises(ValueError):
        configure(['--backend', 'official', '--output', 'example', '--memory-budget-gib', '40',
                   '--exchange-reduce=triton'])
