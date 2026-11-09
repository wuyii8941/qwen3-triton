from pathlib import Path

import pytest

from scripts.campaign_fsdp_exchange_quality import command, selection, audit_campaign


def rows():
    return [dict(repeat=r, case=name, status='passed', median_step_seconds=seconds,
                 peak_allocated_gib=memory) for r in (0, 1) for name, seconds, memory in
            (('torch_decoder_auto_k32', 5.3, 38.6), ('triton_head_simple_k32', 5.7, 40.9))]


def test_selection_has_no_quality_input():
    result = selection(rows())
    assert result['native_keep'] == 36 and result['candidate_keep'] == 32
    assert result['selection_uses_validation_or_test_loss'] is False


@pytest.mark.parametrize('field,value', [('median_step_seconds', 6.), ('peak_allocated_gib', 41.), ('status', 'failed')])
def test_no_pruning_without_two_order_dominance(field, value):
    values = rows()
    values[2][field] = value
    with pytest.raises(ValueError):
        selection(values)


@pytest.mark.parametrize('mode,keep', [('native', '36'), ('decoder', '32')])
def test_full_protocol_no_short_cutoff(mode, keep):
    cmd = command(Path('unused'), mode, Path('training/fsdp_convergence_protocol_v1.json'))
    assert '--tiny' not in cmd and '--preflight-steps' not in cmd
    assert cmd[cmd.index('--keep-full-layers') + 1] == keep
    assert cmd[cmd.index('--memory-budget-gib') + 1] == '42'
    assert cmd[cmd.index('--checkpoint-every') + 1] == '1'
    assert ('--exchange-backend' in cmd) == (mode == 'decoder')


def test_head_not_silently_promoted():
    with pytest.raises(ValueError):
        command(Path('unused'), 'head', Path('protocol'))


def test_partial_campaign_requires_explicit_diagnostic_flag(tmp_path):
    with pytest.raises(ValueError):
        audit_campaign(tmp_path)
