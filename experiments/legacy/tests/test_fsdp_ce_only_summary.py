import json
import pytest
from scripts.summarize_fsdp_ce_only import summarize
from tests.test_fsdp_memory_summary import result


def setup(root):
    cases={'baseline_full_k8': (8,2,'full',0),'ce_full_k12': (12,2,'full',128)}
    for name,policy in cases.items():
        r=result();r['optimization'].update(keep_full_layers=policy[0],checkpoint_every=policy[1],
            checkpoint_mode=policy[2],ce_only_chunk_size=policy[3])
        path=root/name;path.mkdir();(path/'result.json').write_text(json.dumps(r))
        (root/(name+'.log')).write_text('done')
    (root/'status.log').write_text('\n'.join(f'{n} returncode=0' for n in cases)+'\n')
    return cases


def test_explicit_ce_only_comparison(tmp_path):
    report=summarize(tmp_path,setup(tmp_path))
    assert report['complete'] and report['records'][1]['throughput_speedup_vs_baseline']==1


@pytest.mark.parametrize('fault',['label','source','allocator','incomplete','loss','baseline'])
def test_reject_unmatched_or_failed_comparison(tmp_path,fault):
    cases=setup(tmp_path);path=tmp_path/'ce_full_k12/result.json';r=json.loads(path.read_text())
    if fault=='label':r['optimization']['ce_only_chunk_size']=7
    elif fault=='source':r['manifest']['implementation_sources']['sha256']='other'
    elif fault=='allocator':r['allocator']={'backend':'different'}
    elif fault=='loss':r['optimization']['loss_chunk_size']=7
    elif fault=='incomplete':r['complete']=False
    else:(tmp_path/'status.log').write_text('baseline_full_k8 returncode=1\nce_full_k12 returncode=0\n')
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):summarize(tmp_path,cases)
