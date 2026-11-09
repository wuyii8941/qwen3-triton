from scripts.screen_fsdp_exchange_40 import CASES,command,decisions,verify_gates
import pytest


def test_budget_change_is_explicit_not_relabelled_42gib(tmp_path):
    for name,spec in CASES.items():
        cmd=command(spec,tmp_path/name)
        assert cmd[cmd.index('--memory-budget-gib')+1]=='40'
        assert cmd[cmd.index('--keep-full-layers')+1]==str(spec[0])
        assert cmd[cmd.index('--steps')+1]=='12'
        assert '--reference-check' in cmd


def test_strongest_feasible_native_and_matching_protocol_required():
    times=dict(native_auto_k32=6.,native_auto_k34=5.,native_simple_k32=5.5,
        native_simple_k34=4.,torch_decoder_auto_k28=4.5,triton_head_simple_k26=3.5)
    rows=[dict(repeat=r,case=n,status='passed',median_step_seconds=times[n]) for r in (0,1) for n in CASES]
    result=decisions(rows)
    assert not result['torch_decoder_auto_k28']['over_two_percent_both_orders']
    assert result['triton_head_simple_k26']['over_two_percent_both_orders']
    for row in rows:
        if 'native_simple' in row['case']:row['status']='oom'
    assert not decisions(rows)['triton_head_simple_k26']['over_two_percent_both_orders']


def test_missing_numerical_gate_is_not_success(tmp_path):
    with pytest.raises(FileNotFoundError):verify_gates(tmp_path)
