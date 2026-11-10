import hashlib
import json

import pytest

from scripts.audit_fsdp_decoder_backend import audit
from scripts.screen_fsdp_decoder_backend import helpers
from tests.test_screen_fsdp_decoder_backend import gate, text


def test_incomplete_cannot_be_final(tmp_path):
    with pytest.raises(ValueError, match='incomplete'):
        audit(tmp_path)


@pytest.mark.parametrize('corruption', ['log', 'dtype', 'source'])
def test_gate_evidence_corruption_rejected(tmp_path, corruption):
    rows = gate()
    for row in rows:
        row.update(dtype='torch.bfloat16', accumulation=1)
    if corruption == 'dtype':
        rows[0]['dtype'] = 'torch.float32'
    log = text(rows).encode()
    (tmp_path / 'gate.log').write_bytes(log + (b' ' if corruption == 'log' else b''))
    (tmp_path / 'gate.json').write_text(json.dumps(dict(passed=True, records=rows,
        source_sha256={} if corruption == 'source' else helpers(),
        log_sha256=hashlib.sha256(log).hexdigest())))
    with pytest.raises(ValueError, match='gate/source/log'):
        audit(tmp_path, allow_partial=True)
