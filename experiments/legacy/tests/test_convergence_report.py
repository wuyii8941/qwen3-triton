import json

from benchmarks.report_convergence_comparison import audit, events


def setup_run(tmp_path, status):
    (tmp_path / 'frozen_protocol.json').write_text(json.dumps({'plateau': {'min_delta': .002}}))
    for name, seconds in [('native', 100), ('triton', 80)]:
        folder = tmp_path / name
        folder.mkdir()
        row = dict(event='validation', loss=2., step=250, elapsed_seconds=seconds)
        (folder / 'events.jsonl').write_text(json.dumps(row) + '\n')
        if status:
            (folder / 'summary.json').write_text(json.dumps(dict(status=status, elapsed_seconds=seconds)))


def test_incomplete_has_no_speedup(tmp_path):
    setup_run(tmp_path, None)
    result = audit(tmp_path)
    assert not result['finished']
    assert not any('speedup' in k for k in result)


def test_safety_limit_is_not_convergence(tmp_path):
    setup_run(tmp_path, 'not_converged')
    result = audit(tmp_path)
    assert result['finished'] and not result['converged_both']
    assert not any('speedup' in k for k in result)


def test_completed_without_observer_does_not_invent_clock(tmp_path):
    setup_run(tmp_path, 'converged')
    result = audit(tmp_path)
    assert result['main_entry_matched_quality_speedup'] == 1.25
    assert 'external_process_lifetime_speedup' not in result
    assert all(c['external_observed_seconds'] is None for c in result['first_crossings'].values())


def test_concurrent_partial_line(tmp_path):
    path = tmp_path / 'events.jsonl'
    path.write_text('{"event":"train"}\n{"event":')
    assert events(path) == [{'event': 'train'}]


def test_external_clock_ratios_require_actual_observations(tmp_path):
    setup_run(tmp_path, 'converged')
    clocks = {}
    for name, seconds in [('native', 110), ('triton', 90)]:
        row = json.loads((tmp_path / name / 'events.jsonl').read_text())
        observed = dict(job_event=row, process_elapsed_seconds=seconds)
        (tmp_path / name / 'external_clock_events.jsonl').write_text(json.dumps(observed) + '\n')
        clocks[name] = dict(summary_present=True, process_lifetime_seconds=seconds + 10)
    (tmp_path / 'external_process_clocks.json').write_text(json.dumps(clocks))
    result = audit(tmp_path)
    assert result['external_observed_matched_quality_speedup'] == 110 / 90
    assert result['external_process_lifetime_speedup'] == 1.2


def test_quality_target_not_reached_withholds_ratio(tmp_path):
    setup_run(tmp_path, 'converged')
    path = tmp_path / 'triton' / 'events.jsonl'
    row = json.loads(path.read_text())
    row['loss'] = 2.1
    path.write_text(json.dumps(row) + '\n')
    result = audit(tmp_path)
    assert result['first_crossings']['triton']['job_event'] is None
    assert 'main_entry_matched_quality_speedup' not in result
