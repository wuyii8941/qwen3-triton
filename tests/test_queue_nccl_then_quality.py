import os
import sys
import pytest
from scripts.queue_nccl_then_quality import bounded_run,large_message_candidate,is_resume_queue_command,resources_ready


def report():
    return dict(complete=True,simple_over_two_percent_in_both_orders={f'{size}/{op}':False
        for size in ('small','decoder','head') for op in ('all_gather','reduce_scatter')})


def test_only_repeated_large_message_gain_requires_model_followup():
    r=report()
    assert not large_message_candidate(r)
    r['simple_over_two_percent_in_both_orders']['small/all_gather']=True
    assert not large_message_candidate(r)
    r['simple_over_two_percent_in_both_orders']['decoder/all_gather']=True
    assert large_message_candidate(r)


def test_incomplete_diagnostic_is_not_a_speedup():
    with pytest.raises(ValueError):large_message_candidate(dict(complete=False))


def test_both_explicit_resume_queue_entrypoints_are_recognized():
    assert is_resume_queue_command(b'python -m scripts.queue_real_fsdp_resume --output x')
    assert is_resume_queue_command(b'python -m scripts.queue_existing_fsdp_resume --output x')
    assert not is_resume_queue_command(b'python other_job.py')


def test_probe_budget_is_distinct_from_full_model_budget():
    readings=[(3000,45000,0)]*4
    assert resources_ready(readings,communication=True)
    assert not resources_ready(readings)
    assert resources_ready([(1000,47000,0)]*4)


@pytest.mark.parametrize('readings',[
    [(3000,31000,0)]*4,[(3000,45000,10)]*4,[(3000,45000,0)]*3])
def test_probe_still_requires_free_capacity_and_low_utilization(readings):
    assert not resources_ready(readings,communication=True)


def test_bounded_success(tmp_path):
    code,timed_out=bounded_run([sys.executable,'-c','print("probe")'],tmp_path/'ok.log',dict(os.environ))
    assert code==0 and not timed_out
    assert (tmp_path/'ok.log').read_text().strip()=='probe'


def test_bounded_timeout_stops_owned_session(tmp_path):
    code,timed_out=bounded_run([sys.executable,'-c','import time; time.sleep(60)'],
        tmp_path/'timeout.log',dict(os.environ),timeout=.1,grace_seconds=2)
    assert timed_out and code!=0
