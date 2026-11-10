from scripts.verify_fsdp_scratch_bounded import collect_idle


def test_three_consecutive_samples_after_busy_reset():
    ready = [[8192, 9] for _ in range(4)]
    busy = [[8192, 10] for _ in range(4)]
    rows = iter([ready, ready, busy, ready, ready, ready])
    pauses = []
    assert collect_idle(lambda: next(rows), attempts=6, pause=pauses.append) == [ready] * 3
    assert pauses == [1] * 5


def test_failure_bounded_and_threshold_not_weakened():
    for bad in ([[8191, 0]] * 4, [[8192, 10]] * 4, [[8192, 0]] * 3):
        pauses = []
        assert collect_idle(lambda: bad, attempts=4, pause=pauses.append) is None
        assert pauses == [1] * 3
