import json

import pytest

from scripts.audit_fsdp_retention_pressure import audit, audit_resources


def resources():
    return dict(allocator_budget_gib=42, free_margin_gib=2.5, idle_confirmations=3,
        samples=[dict(readings=[(0, 48000, 0)] * 4,
                      mem_available_bytes=170 * 2**30, disk_free_bytes=20 * 2**30) for _ in range(3)])


def test_resource_budget_evidence():
    audit_resources(resources())


@pytest.mark.parametrize('field,value', [('allocator_budget_gib', 44), ('free_margin_gib', 1),
                                       ('idle_confirmations', 1)])
def test_wrong_resource_contract_rejected(field, value):
    row = resources()
    row[field] = value
    with pytest.raises(ValueError, match='Resource'):
        audit_resources(row)


def test_all_three_resource_samples_required():
    row = resources()
    row['samples'][2]['readings'][3] = (10000, 39000, 50)
    with pytest.raises(ValueError, match='Resource'):
        audit_resources(row)


def test_incomplete_screen_cannot_be_final(tmp_path):
    with pytest.raises(ValueError, match='incomplete'):
        audit(tmp_path)


def test_reordered_partial_rejected(tmp_path):
    (tmp_path / 'progress.json').write_text(json.dumps([dict(repeat=0, case='decoder_k24')]))
    with pytest.raises(ValueError, match='ABCD/DCBA'):
        audit(tmp_path, allow_partial=True)
