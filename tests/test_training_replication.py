import json
import hashlib

from benchmarks.run_training_replication import gate_preflight, schedule
from benchmarks.summarize_training_replication import summarize
import pytest


def test_schedule_balances_backend_order():
    rows=schedule()
    assert len(rows)==12
    for backend in ('native','compile','triton'):
        assert sorted(i%3 for i,r in enumerate(rows[:9]) if r['backend']==backend)==[0,1,2]
    assert len({r['seed'] for r in rows if r['seq']==512})==3


def test_empty_plan_is_incomplete_not_fast(tmp_path):
    (tmp_path/'plan.json').write_text(json.dumps(dict(jobs=schedule())))
    report=summarize(tmp_path)
    assert not report['all_jobs_terminated'] and not report['all_jobs_converged']
    assert all(r['n']==0 and r['geometric_mean'] is None for r in report['aggregate'])


def test_gate_rejects_bad_compile_loss(tmp_path):
    for backend in ('native','compile','triton'):
        directory=tmp_path/f'preflight_s512_{backend}';directory.mkdir()
        row=dict(event='validation',step=2,loss=3.1 if backend=='compile' else 3.)
        (directory/'events.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(RuntimeError,match='preflight mismatch'):
        gate_preflight(tmp_path,512)


def test_gate_requires_identical_observed_steps(tmp_path):
    for backend in ('native','compile','triton'):
        directory=tmp_path/f'preflight_s512_{backend}';directory.mkdir()
        row=dict(event='validation',step=4 if backend=='compile' else 2,loss=3.)
        (directory/'events.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(RuntimeError,match='identical steps'):
        gate_preflight(tmp_path,512)


def make_three_jobs(root, status='converged', candidate_loss=2.001):
    rows=[dict(seq=512,seed=1,backend=b) for b in ('native','compile','triton')]
    (root/'plan.json').write_text(json.dumps(dict(jobs=rows)))
    for b,wall,cross in [('native',100,80),('compile',80,60),('triton',90,70)]:
        name=f's512_seed1_{b}';directory=root/name;directory.mkdir()
        protocol=root/f'{name}.protocol.json'
        protocol.write_text(json.dumps(dict(plateau=dict(min_delta=.002))))
        summary=dict(status=status if b=='triton' else 'converged',
            protocol_hash=hashlib.sha256(protocol.read_bytes()).hexdigest(),test_loss=2.)
        (directory/'summary.json').write_text(json.dumps(summary))
        event=dict(event='validation',step=250,loss=2. if b=='native' else candidate_loss)
        (directory/'events.jsonl').write_text(json.dumps(event)+'\n')
        clock=[dict(event='observed',process_elapsed_seconds=cross,job_event=event),
               dict(event='exit',returncode=0,wall_seconds=wall)]
        (root/f'{name}.clock.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in clock))


def test_report_compares_directly_against_compile(tmp_path):
    make_three_jobs(tmp_path)
    result=summarize(tmp_path)
    direct=next(p for p in result['comparisons'] if p['candidate']=='triton_vs_compile')
    assert result['all_jobs_converged']
    assert direct['matched_quality_speedup']==60/70  # Triton loses; preserve it.


def test_safety_limited_pair_has_no_speedup(tmp_path):
    make_three_jobs(tmp_path,status='not_converged')
    result=summarize(tmp_path)
    assert result['all_jobs_terminated'] and not result['all_jobs_converged']
    assert all(not any('speedup' in key for key in p)
               for p in result['comparisons'] if 'triton' in p['candidate'])


def test_unreached_quality_target_has_no_matched_ratio(tmp_path):
    make_three_jobs(tmp_path,candidate_loss=2.1)
    result=summarize(tmp_path)
    assert all(not any('matched_quality_speedup' in key for key in p) for p in result['comparisons'])
