from scripts.report_bf16_exchange import candidates


def test_torch_candidate_is_not_discarded():
    report = {'paired_throughput_ratios': {
        'decoder': {'exchange_torch': [1.2, 1.3], 'exchange_triton': [.9, 1.4]},
        'head': {'exchange_torch': [1.1, 1.01], 'exchange_triton': [1.2, 1.4]},
        'small': {'exchange_triton': [2., 2.]}}}
    assert [(r['size'], r['backend']) for r in candidates(report)] == [
        ('decoder', 'exchange_torch'), ('head', 'exchange_triton')]


def test_threshold_is_strict_and_requires_two_orders():
    assert not candidates({'paired_throughput_ratios': {
        'decoder': {'a': [1.02, 1.3], 'b': [1.5], 'c': [1.2, float('nan')]}}})
