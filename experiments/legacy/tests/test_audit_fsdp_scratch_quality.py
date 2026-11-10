import pytest

from scripts.audit_fsdp_scratch_quality import full_comparison


def job(seconds=10., loss=2.):
    return dict(complete=True, converged=True, stop_reason='validation_plateau', best_parameters_restored=True,
                elapsed_seconds=seconds, training=[dict(seconds=1.), dict(seconds=2.)], test=dict(loss=loss))


def test_full_jobs_report_observation_not_equivalence():
    result = full_comparison(job(), job(8., 2.001))
    assert result['bounded_over_native_job_time'] == .8
    assert result['native_median_recorded_step_seconds'] == 1.5
    assert 'not statistical quality equivalence' in result['scope']


@pytest.mark.parametrize('field,value', [('complete', False), ('converged', False),
    ('stop_reason', 'safety_deadline'), ('best_parameters_restored', False)])
def test_cutoffs_or_incomplete_jobs_never_qualify(field, value):
    candidate = job()
    candidate[field] = value
    with pytest.raises(ValueError):
        full_comparison(job(), candidate)
