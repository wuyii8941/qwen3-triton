import pytest
from scripts.report_fsdp_convergence import summarize,compare_jobs


def test_running_improvement_is_not_convergence():
    s=summarize(dict(backend='official',complete=False,converged=False,
        validation=[dict(loss=3.),dict(loss=2.)],training=[{}]))
    assert not s['converged'] and s['best_observed_validation']==2.
    assert s['heldout_test'] is None


def test_plateau_uses_cumulative_significant_improvement():
    s=summarize(dict(backend='official',complete=False,converged=False,
        protocol=dict(plateau=dict(min_delta=.002,patience=8,min_checks=10)),
        validation=[dict(loss=x) for x in [2.,1.999,1.997,1.9965]]))
    assert s['plateau']['reference_loss']==1.997
    assert s['plateau']['stale_checks']==1
    assert not s['plateau']['criterion_reached']
    assert not s['converged']


@pytest.mark.parametrize('complete,reason',[(False,'validation_plateau'),(True,'safety_deadline')])
def test_rejects_false_convergence(complete,reason):
    with pytest.raises(ValueError):summarize(dict(backend='official',complete=complete,
        converged=True,stop_reason=reason))


def test_comparison_waits_for_both_jobs():
    r=dict(backend='official',protocol={'seed':1},complete=False,converged=False)
    assert compare_jobs(r,r)==dict(ready=False,reason='Both jobs must finish at validation plateau')


def test_comparison_reports_quality_separately_from_job_time():
    r=dict(backend='official',protocol=dict(seed=1,plateau=dict(min_delta=.002,patience=1,min_checks=2)),
        validation=[dict(loss=2.),dict(loss=2.)],complete=True,converged=True,
        stop_reason='validation_plateau',best_parameters_restored=True,
        elapsed_seconds=10.,best_validation=2.,test=dict(loss=2.1,tokens=100,blocks=2))
    candidate=dict(r,backend='custom_avg',elapsed_seconds=12.,best_validation=2.01)
    result=compare_jobs(r,candidate)
    assert result['ready'] and result['candidate_over_baseline_job_time']==1.2
    assert result['candidate_minus_baseline_best_validation']==pytest.approx(.01)
    with pytest.raises(ValueError):compare_jobs(r,dict(candidate,best_parameters_restored=False))
    with pytest.raises(ValueError):compare_jobs(r,dict(candidate,protocol={'seed':2}))
    with pytest.raises(ValueError):compare_jobs(r,dict(candidate,validation=[dict(loss=2.)]))
