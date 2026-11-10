import json
from pathlib import Path
import pytest
from scripts.audit_fsdp_resume import audit


@pytest.fixture
def records(tmp_path):
    source=Path('artifacts/fsdp_real_resume_pipeline_tiny_v1')
    if not source.exists():pytest.skip('Local GPU-generated replay fixture absent')
    for name in ['result.json',*[f'rank{r}_result.json' for r in range(4)]]:
        (tmp_path/name).write_bytes((source/name).read_bytes())
    return tmp_path


def test_real_gpu_tiny_records_cannot_prove_8b(records):
    r=audit(records)
    assert r['passed'] and r['synthetic_tiny'] and not r['full_8b_resume_proven']
    assert len(r['records'])==4


@pytest.mark.parametrize('bad',['rank','moment','source','count','rng','nonfinite'])
def test_reject_incomplete_all_rank_evidence(records,bad):
    path=records/'rank3_result.json';r=json.loads(path.read_text())
    if bad=='rank':r['rank']=2
    elif bad=='moment':r['state']['optimizer_states'].pop(next(iter(r['state']['parameters'])))
    elif bad=='source':r['source_sha256']='0'*64
    elif bad=='count':r['resources']['parameters']+=1
    elif bad=='rng':r['rng'].pop('torch_cuda')
    elif bad=='nonfinite':r['replay_updates'][0]['loss']=float('nan')
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):audit(records)


def test_reject_tiny_relabel_as_full(records):
    path=records/'result.json';r=json.loads(path.read_text())
    r['synthetic_tiny']=False;r['full_8b_resume_proven']=True
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):audit(records)


def test_existing_replay_uses_explicit_read_only_guard_not_new_save_budget(records):
    path=records/'result.json';r=json.loads(path.read_text())
    r.update(wrote_checkpoint=False,reader='offset_sorted');path.write_text(json.dumps(r))
    for rank in range(4):
        path=records/f'rank{rank}_result.json';row=json.loads(path.read_text())
        row.update(wrote_checkpoint=False,reader='offset_sorted')
        row['resources'].update(resources_sufficient=False,read_only_replay_resources_sufficient=True)
        path.write_text(json.dumps(row))
    assert audit(records)['passed']
    row['reader']='native';path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='guard/reader identity'):audit(records)
