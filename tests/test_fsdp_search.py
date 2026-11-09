import copy
import pytest
from training.fsdp_search import retention_candidates,select_retention


def row(k,seconds):
    return dict(keep_full_layers=k,status='passed',result=dict(complete=True,
        model='qwen',world_size=4,batch=1,seq=512,steps=6,warmup=3,
        precision='fp32/bf16',attention='flash',memory_budget_gib=42.,
        manifest=dict(input_sha256='same'),optimization=dict(keep_full_layers=k),
        median_step_seconds=seconds,records=[dict(peak_allocated_gib=35.)],
        initial_hf_parity=dict(max_abs=0.)))


def test_candidates_include_capacity_endpoints():
    assert retention_candidates(36)==[0,4,8,12,16,24,36]
    assert retention_candidates(2)==[0,2]


def test_oom_is_not_winner_and_baseline_required():
    rows=[row(0,7.),row(8,6.5),dict(keep_full_layers=36,status='oom')]
    assert select_retention(rows,42.)['best_keep_full_layers']==8
    with pytest.raises(ValueError):select_retention(rows[1:],42.)


@pytest.mark.parametrize('fault',['inputs','workload','budget','nan','parity','incomplete','memory','label','accumulation','sync','checkpoint','loss','sources'])
def test_reject_invalid_evidence(fault):
    rows=[row(0,7.),row(8,6.5)];r=rows[1]['result']
    if fault=='inputs':r['manifest']['input_sha256']='different'
    elif fault=='sources':r['manifest']['implementation_sources']={'sha256':'different'}
    elif fault=='workload':r['batch']=2
    elif fault=='budget':r['memory_budget_gib']=43.
    elif fault=='nan':r['median_step_seconds']=float('nan')
    elif fault=='parity':r.pop('initial_hf_parity')
    elif fault=='incomplete':r['complete']=False
    elif fault=='memory':r['records'][0]['peak_allocated_gib']=43.
    elif fault=='accumulation':r['accumulation']=2
    elif fault=='sync':r['gradient_sync']='defer'
    elif fault=='checkpoint':r['optimization']['checkpoint_every']=2
    elif fault=='loss':r['optimization']['loss_chunk_size']=128
    else:r['optimization']['keep_full_layers']=4
    with pytest.raises(ValueError):select_retention(rows,42.)


def test_non_oom_errors_and_duplicates_are_not_silently_skipped():
    with pytest.raises(ValueError):select_retention([row(0,7.),dict(keep_full_layers=8,status='failed')],42.)
    with pytest.raises(ValueError):select_retention([row(0,7.),copy.deepcopy(row(0,7.))],42.)
