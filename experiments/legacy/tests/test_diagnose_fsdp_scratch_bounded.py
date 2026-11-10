from pathlib import Path

import pytest

from benchmarks.deterministic_fsdp_scratch_bounded import require_environment
from scripts.diagnose_fsdp_scratch_bounded import command


@pytest.mark.parametrize('env', [{}, {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}])
def test_explicit_cublas_configuration_required(env):
    with pytest.raises(ValueError):
        require_environment(env)


@pytest.mark.parametrize('candidate', [False, True])
def test_real_four_update_diagnostic(candidate):
    cmd = command(candidate, Path('example'))
    assert '--tiny' not in cmd and '--reference-check' in cmd
    assert cmd[cmd.index('--warmup') + 1] == '1'
    assert cmd[cmd.index('--steps') + 1] == '3'
    assert cmd[cmd.index('--memory-budget-gib') + 1] == '40'
    assert ('--bounded-exchange-scope' in cmd) == candidate
