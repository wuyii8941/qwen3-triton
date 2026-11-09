import pytest

from benchmarks.compare_fsdp_scratch_bounded import configure


def test_native_path_does_not_enable_adapter():
    args, base = configure(['--output', 'example', '--backend', 'official'])
    assert args.bounded_exchange_scope is None
    assert not any('exchange-' in arg for arg in base)


@pytest.mark.parametrize('scope', ['head', 'decoder', 'all'])
def test_scope_passed_explicitly(scope):
    args, base = configure(['--output', 'example', '--backend', 'official',
                            '--bounded-exchange-scope', scope, '--memory-budget-gib', '40'])
    assert args.bounded_exchange_scope == scope
    assert base[-4:] == ['--exchange-reduce', 'triton', '--exchange-scope', scope]
    assert base[base.index('--memory-budget-gib') + 1] == '40'


@pytest.mark.parametrize('flags', [['--exchange-reduce', 'triton'],
                                  ['--exchange-scope=head'], ['--exchange-reduce=triton']])
def test_no_silent_original_adapter(flags):
    with pytest.raises(ValueError):
        configure(['--output', 'example', '--backend', 'official', *flags])
