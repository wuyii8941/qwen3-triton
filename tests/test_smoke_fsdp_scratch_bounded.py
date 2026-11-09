from pathlib import Path

import pytest

from scripts.smoke_fsdp_scratch_bounded import command


@pytest.mark.parametrize('scope', [None, 'head', 'decoder'])
def test_tiny_smoke_does_not_request_pretrained_reference(scope):
    cmd = command(scope, Path('example'))
    assert '--tiny' in cmd and '--reference-check' not in cmd
    assert cmd[cmd.index('--memory-budget-gib') + 1] == '8'
    if scope:
        assert cmd[-2:] == ['--bounded-exchange-scope', scope]
    else:
        assert '--bounded-exchange-scope' not in cmd
