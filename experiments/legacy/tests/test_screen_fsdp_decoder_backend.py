import json

import pytest

from scripts.screen_fsdp_decoder_backend import ORDER, command, audit_gate, selection


@pytest.mark.parametrize('name', ORDER)
def test_only_local_backend_varies(name):
    cmd = command(name, 'unused')
    assert cmd[cmd.index('--keep-full-layers') + 1] == '32'
    assert cmd[cmd.index('--memory-budget-gib') + 1] == '41.875'
    assert cmd[cmd.index('--exchange-reduce') + 1] == name.split('_')[0]
    assert cmd[cmd.index('--exchange-scope') + 1] == 'decoder'
    assert '--profile' not in cmd


def gate():
    return [dict(passed=True, tied=tied, backend='official', world=4, exchange_reduce='triton',
        head_dim=128, seq=129, steps=3, keep_full_layers=2, checkpoint_every=1,
        deterministic=True, exchange_callback_calls=12 if tied else 15) for tied in (False, True)]


def text(rows):
    return '\n'.join(json.dumps(row) for row in rows)


def test_both_gate_variants_required():
    assert len(audit_gate(text(gate()))) == 2
    with pytest.raises(ValueError):
        audit_gate(text(gate()[:1]))


@pytest.mark.parametrize('field,value', [('head_dim', 16), ('seq', 11), ('steps', 1),
                                       ('exchange_reduce', 'torch'), ('deterministic', False)])
def test_wrong_gate_contract_rejected(field, value):
    rows = gate()
    rows[1][field] = value
    with pytest.raises(ValueError):
        audit_gate(text(rows))


def test_one_order_cannot_qualify_candidate():
    rows = [dict(case=name, median_step_seconds=5 if name.startswith('triton') else 6) for name in ORDER]
    assert selection(rows)['worth_quality_validation']
    rows[2]['median_step_seconds'] = 7
    assert not selection(rows)['worth_quality_validation']
