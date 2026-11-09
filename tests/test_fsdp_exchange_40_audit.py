import json
import pytest
from scripts.audit_fsdp_exchange_40 import audit,audit_resources


def write_resource(path,*,budget=40,free=43520,util=0):
    path.write_text(json.dumps(dict(allocator_budget_gib=budget,free_margin_gib=2.5,idle_confirmations=3,
        samples=[dict(unix_seconds=100+15*i,readings=[[4000,free,util] for _ in range(4)]) for i in range(3)])))


def test_resource_policy_does_not_mix_budgets(tmp_path):
    path=tmp_path/'case_resources.json';write_resource(path)
    assert len(audit_resources(path))==64
    write_resource(path,budget=42)
    with pytest.raises(ValueError):audit_resources(path)


@pytest.mark.parametrize('free,util',[(43519,0),(43520,10)])
def test_insufficient_margin_or_busy_gpu_rejected(tmp_path,free,util):
    path=tmp_path/'case_resources.json';write_resource(path,free=free,util=util)
    with pytest.raises(ValueError):audit_resources(path)


def test_incomplete_study_needs_explicit_partial_mode(tmp_path):
    with pytest.raises(ValueError,match='incomplete'):audit(tmp_path)
