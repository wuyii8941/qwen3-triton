from scripts.audit_fsdp_scratch_deterministic import compare_runs


def run(values):
    return {'records': [{'loss': value} for value in values]}


def test_exact_native_and_candidate_within_existing_threshold():
    runs = dict(native_a=run([1., 2., 3., 4.]), native_b=run([1., 2., 3., 4.]),
                bounded_decoder=run([1., 2.000001, 3., 4.]))
    _, _, exact, eligible = compare_runs(runs)
    assert exact and eligible


def test_even_small_native_instability_blocks_preflight():
    runs = dict(native_a=run([1., 2., 3., 4.]), native_b=run([1., 2.000001, 3., 4.]),
                bounded_decoder=run([1., 2., 3., 4.]))
    native, _, exact, eligible = compare_runs(runs)
    assert native['within_declared_loss_tolerance'] and not exact and not eligible


def test_candidate_exceeding_unchanged_threshold_blocks_preflight():
    runs = dict(native_a=run([1., 2., 3., 4.]), native_b=run([1., 2., 3., 4.]),
                bounded_decoder=run([1., 2.01, 3., 4.]))
    _, candidate, exact, eligible = compare_runs(runs)
    assert exact and not eligible
    assert not candidate['native_a']['within_declared_loss_tolerance']
