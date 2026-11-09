import json
from pathlib import Path

import pytest

from scripts.campaign_fsdp_scratch_quality import audit_job, command, select_keep


def report():
    return dict(complete=True, decisions={'bounded_decoder_k34': {'over_two_percent_both_orders': True}},
                records=[dict(case=f'r{r}_native_k{k}', status='passed', median_step_seconds=t)
                         for r in (0, 1) for k, t in ((32, 6.1), (34, 5.9))])


def test_select_native_without_test_loss():
    assert select_keep(report()) == 34


def test_capacity_failure_not_fast_native():
    raw = report()
    raw['records'][-1]['status'] = 'oom'
    assert select_keep(raw) == 32


def test_performance_gate_cannot_be_bypassed():
    raw = report()
    raw['decisions']['bounded_decoder_k34']['over_two_percent_both_orders'] = False
    with pytest.raises(ValueError):
        select_keep(raw)


@pytest.mark.parametrize('bounded', [False, True])
def test_preflight_explicit_budget_and_same_policy(bounded):
    cmd = command(Path('example'), keep=34, bounded=bounded, protocol=Path('preflight.json'), preflight=True)
    assert cmd[cmd.index('--memory-budget-gib') + 1] == '40'
    assert cmd[cmd.index('--keep-full-layers') + 1] == '34'
    assert cmd[cmd.index('--checkpoint-every') + 1] == '1'
    assert cmd[cmd.index('--protocol') + 1] == 'preflight.json'
    assert cmd[cmd.index('--preflight-steps') + 1] == '2'
    assert ('--bounded-exchange-scope' in cmd) == bounded


def test_long_job_no_shortcut_or_tiny():
    cmd = command(Path('example'), keep=34, bounded=True, protocol=Path('frozen.json'))
    assert '--tiny' not in cmd and '--preflight-steps' not in cmd
    with pytest.raises(ValueError):
        command(Path('example'), keep=34, bounded=True, protocol=Path('frozen.json'), tiny=True)


def test_missing_post_update_validation_rejected(tmp_path):
    protocol = json.loads(Path('training/fsdp_convergence_protocol_v1.json').read_text())
    protocol['validation']['every_updates'] = 2
    raw = dict(complete=True, backend='official', protocol=protocol, best_parameters_restored=True,
               optimization=dict(requested_keep_full_layers=34, keep_full_layers=34,
                                 checkpoint_every=1, checkpoint_mode='full'),
               converged=False, stop_reason='preflight_cutoff',
               training=[dict(step=1), dict(step=2)], validation=[dict(step=0)])
    (tmp_path / 'result.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='post-update'):
        audit_job(tmp_path, keep=34, bounded=False, protocol=protocol,
                  protocol_file_sha256='unused', preflight=True)
