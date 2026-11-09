import copy
import json
import pytest
from scripts.summarize_fsdp_memory import CASES,audit_result,summarize


def result():
    rows=[dict(step=i+1,warmup=i<3,max_rank_seconds=6.,loss=2.,
        peak_allocated_gib=39.,peak_reserved_gib=42.) for i in range(9)]
    return dict(complete=True,backend='official',model='qwen',world_size=4,batch=1,seq=512,
        warmup=3,steps=6,precision='FP32/BF16',attention='Flash',memory_budget_gib=42.,
        records=rows,median_step_seconds=6.,global_tokens_per_second=2048/6,
        initial_hf_parity=dict(max_abs=0.),optimization=dict(keep_full_layers=24,
            checkpoint_every=0,checkpoint_mode='full'),
        manifest=dict(input_sha256='inputs',implementation_sources=dict(sha256='sources'),
            pytorch='2.10',cuda='12.8',gpu='A6000'))


def test_only_checkpoint_and_retention_policy_can_vary():
    a=result(); b=copy.deepcopy(a)
    b['optimization'].update(keep_full_layers=36,checkpoint_every=2)
    assert audit_result(a,(24,0,'full'))==audit_result(b,(36,2,'full'))


def test_communication_candidate_not_silently_treated_as_native():
    r=result();r['optimization']['exchange_reduce']='torch'
    with pytest.raises(ValueError,match='Exchange'):
        audit_result(r,(24,0,'full'))
    audit_result(r,(24,0,'full'),allow_exchange=True)


@pytest.mark.parametrize('fault',['incomplete','label','accumulation','swiglu','loss','parity','sources',
    'rows','warmup','nan','median','throughput','memory','reserved_memory'])
def test_reject_invalid_evidence(fault):
    r=result()
    if fault=='incomplete':r['complete']=False
    elif fault=='label':r['optimization']['checkpoint_every']=2
    elif fault=='accumulation':r['accumulation']=2
    elif fault=='swiglu':r['optimization']['swiglu']='triton'
    elif fault=='loss':r['optimization']['loss_chunk_size']=128
    elif fault=='parity':r.pop('initial_hf_parity')
    elif fault=='sources':r['manifest'].pop('implementation_sources')
    elif fault=='rows':r['records'].pop()
    elif fault=='warmup':r['records'][4]['warmup']=True
    elif fault=='nan':r['records'][4]['loss']=float('nan')
    elif fault=='median':r['median_step_seconds']=5.
    elif fault=='throughput':r['global_tokens_per_second']=1.
    elif fault=='memory':r['records'][4]['peak_allocated_gib']=43.
    elif fault=='reserved_memory':r['records'][4]['peak_reserved_gib']=43.
    with pytest.raises(ValueError):audit_result(r,(24,0,'full'))


def write_screen(root,cases=CASES):
    lines=[]
    for name,policy in cases.items():
        if name=='full_k36':
            (root/(name+'.log')).write_text('torch.OutOfMemoryError: CUDA out of memory')
            lines.append(name+' returncode=1');continue
        r=result();r['optimization'].update(dict(zip(
            ('keep_full_layers','checkpoint_every','checkpoint_mode'),policy)))
        path=root/name;path.mkdir()
        (path/'result.json').write_text(json.dumps(r))
        (root/(name+'.log')).write_text('worker completed')
        lines.append(name+' returncode=0')
    (root/'status.log').write_text('\n'.join(lines)+'\n')


def test_complete_screen_keeps_oom_out_of_speed_ranking(tmp_path):
    write_screen(tmp_path)
    r=summarize(tmp_path)
    assert r['complete'] and r['screening_winner']=='native_k24'
    assert r['records'][-1]['status']=='oom'
    assert all('sha256' in x for x in r['records'] if x['status']=='passed')


@pytest.mark.parametrize('keep', [0,16,20])
def test_explicit_feasible_retention_baseline(tmp_path,keep):
    baseline=f'native_k{keep}'
    cases={baseline: (keep,0,'full'), 'full_k16': (16,2,'full')}
    write_screen(tmp_path,cases)
    report=summarize(tmp_path,cases,baseline_case=baseline)
    assert report['baseline_case']==baseline
    assert all(r['throughput_speedup_vs_baseline']==1 for r in report['records'])
    assert all('throughput_speedup_vs_native_k24' not in r for r in report['records'])
    with pytest.raises(ValueError): summarize(tmp_path,cases)


def test_shape_mismatch_cannot_be_called_speedup(tmp_path):
    cases={'native_k0': (0,0,'full'), 'full_k16': (16,2,'full')}
    write_screen(tmp_path,cases)
    path=tmp_path/'full_k16/result.json'; changed=json.loads(path.read_text())
    changed['seq']=1024
    changed['global_tokens_per_second']*=2
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError): summarize(tmp_path,cases,baseline_case='native_k0')


def test_checkpoint_baseline_is_explicit_not_mislabeled_native(tmp_path):
    cases={'full_k0': (0,2,'full'),'native_k0': (0,0,'full'),'full_k16': (16,2,'full')}
    write_screen(tmp_path,cases)
    # Capacity failure is not an infinite speedup: only passing policies get ratios.
    status=tmp_path/'status.log'
    status.write_text(status.read_text().replace('native_k0 returncode=0','native_k0 returncode=1'))
    (tmp_path/'native_k0.log').write_text('torch.OutOfMemoryError: CUDA out of memory')
    report=summarize(tmp_path,cases,baseline_case='full_k0')
    assert report['baseline_case']=='full_k0'
    assert report['records'][1]['status']=='oom'
    assert 'throughput_speedup_vs_baseline' not in report['records'][1]
    assert all('throughput_speedup_vs_native_k24' not in r for r in report['records'])
    with pytest.raises(ValueError):summarize(tmp_path,cases)


@pytest.mark.parametrize('fault',['sources','non_capacity','incomplete','baseline_oom','allocator'])
def test_screen_rejects_invalid_round(tmp_path,fault):
    write_screen(tmp_path)
    if fault in ('sources','allocator'):
        p=tmp_path/'full_k24/result.json';r=json.loads(p.read_text())
        if fault=='sources':r['manifest']['implementation_sources']['sha256']='changed'
        else:r['allocator']=dict(backend='native',environment=dict(PYTORCH_ALLOC_CONF='expandable_segments:True'))
        p.write_text(json.dumps(r))
    elif fault=='non_capacity':(tmp_path/'full_k36.log').write_text('numerical assertion failed')
    elif fault=='incomplete':
        p=tmp_path/'status.log';p.write_text('\n'.join(p.read_text().splitlines()[:-1]))
    else:
        p=tmp_path/'status.log';p.write_text(p.read_text().replace('native_k24 returncode=0','native_k24 returncode=1'))
        (tmp_path/'native_k24.log').write_text('CUDA out of memory')
    with pytest.raises(ValueError):summarize(tmp_path)


def test_explicit_refinement_policies_are_audited(tmp_path):
    policies=dict(native_k24=[24,0,'full'],full_k35=[35,2,'full'])
    write_screen(tmp_path,policies)
    assert summarize(tmp_path,policies)['complete']
    with pytest.raises(ValueError):summarize(tmp_path,dict(full_k35=[35,2,'full']))
