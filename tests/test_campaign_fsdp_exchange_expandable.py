from pathlib import Path

import pytest

from scripts.campaign_fsdp_exchange_expandable import chosen_policy, audit, command, environment, helpers


def rows():
    return [dict(repeat=r, case=name, status='passed', median_step_seconds=seconds, peak_allocated_gib=memory)
            for r in (0, 1) for name, seconds, memory in
            (('torch_decoder_auto_k32', 5.3, 38.6), ('triton_head_simple_k32', 5.7, 40.9))]


def test_selection_explicit_effective_cap_and_no_quality_selection():
    value = chosen_policy(rows())
    assert value['effective_allocator_cap_gib'] == 41.875
    assert value['audited_reserved_upper_bound_gib'] == value['memory_budget_gib'] == 42
    assert not value['selection_uses_validation_or_test_loss']


@pytest.mark.parametrize('mode,keep', [('native', '36'), ('decoder', '32')])
def test_both_full_jobs_share_allocator_and_original_protocol(mode, keep):
    cmd = command(Path('unused'), mode, Path('training/fsdp_convergence_protocol_v1.json'), preflight=False)
    assert '--tiny' not in cmd and '--preflight-steps' not in cmd
    assert 'benchmarks.fsdp_exchange_quality_expandable' in cmd
    assert cmd[cmd.index('--keep-full-layers') + 1] == keep
    assert environment()['PYTORCH_ALLOC_CONF'] == 'backend:native,expandable_segments:True'


def test_helper_guard_includes_actual_allocator_and_auditors():
    values = helpers()
    for name in ('benchmarks/fsdp_exchange_quality_expandable.py', 'scripts/campaign_fsdp_exchange_expandable.py',
                 'scripts/audit_fsdp_exchange_expandable.py', 'scripts/audit_fsdp_exchange_expandable_preflights.py'):
        assert len(values[name]) == 64


def test_partial_never_implies_full_quality(tmp_path):
    with pytest.raises(ValueError):
        audit(tmp_path)
