import copy
import pytest
from scripts.summarize_fsdp_comparison import summarize


def matrix():
    rows=[]
    for backend in ('custom','custom_avg','official','official_checkpoint'):
        rows.append(dict(backend=backend,returncode=0,result=dict(complete=True,model='same',seq=512,
            batch=1,world_size=4,steps=12,warmup=3,precision='same',attention='same',
            manifest=dict(input_sha256='same'),median_step_seconds=7.,records=[dict(peak_allocated_gib=35.)])))
    return dict(complete=True,records=rows)


def test_accepts_matched_round():
    assert summarize(matrix())['custom_speedup_vs_best_official']==1


@pytest.mark.parametrize('fault',['incomplete','input','failed','loss','sources'])
def test_rejects_invalid_evidence(fault):
    m=copy.deepcopy(matrix())
    if fault=='incomplete':m['complete']=False
    elif fault=='input':m['records'][0]['result']['manifest']['input_sha256']='different'
    elif fault=='sources':m['records'][0]['result']['manifest']['implementation_sources']={'sha256':'different'}
    elif fault=='loss':m['records'][0]['result']['optimization']={'loss_chunk_size':128}
    else:m['records'][0]['returncode']=1
    with pytest.raises(ValueError):summarize(m)
