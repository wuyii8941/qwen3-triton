import pytest
from scripts.queue_fsdp_final_quality import select_quality_policy


def report(k32=(6.1,6.1),k36=(6.2,6.2)):
    policies=dict(interval_k35=(35,2,'full'),all_k32=(32,1,'full'),all_k36=(36,1,'full'))
    times=dict(interval_k35=(6.,6.),all_k32=k32,all_k36=k36)
    return dict(spec=dict(policies=policies),records=[dict(case=f'r{r}_{n}',median_step_seconds=t[r])
        for n,t in times.items() for r in (0,1)])


def test_slower_new_policies_do_not_replace_old_winner():
    assert select_quality_policy(report())['candidate']=='interval_k35'


def test_one_order_only_improvement_is_not_adopted():
    assert select_quality_policy(report(k32=(5.5,6.1)))['candidate']=='interval_k35'


def test_stable_faster_policy_selected_without_test_loss():
    result=select_quality_policy(report(k32=(5.8,5.8),k36=(5.6,5.6)))
    assert result['candidate']=='all_k36'
    assert not result['selection_uses_test_loss']


def test_unexpected_policy_set_rejected():
    r=report();r['spec']['policies'].pop('all_k36')
    with pytest.raises(ValueError):select_quality_policy(r)
