import json
import pytest

from scripts.summarize_fsdp_shape_confirmation import summarize
from tests.test_fsdp_memory_summary import result


def setup(root):
    policies={'native_k24': (24,0,'full'), 'full_k28': (28,2,'full')}
    sample=result()
    workload={k: sample[k] for k in ('model','world_size','batch','seq','warmup','steps',
                                    'memory_budget_gib','precision','attention')}
    workload.update(seq=1024, steps=12)
    spec=dict(policies=policies,candidate='full_k28',workload=workload)
    order=['r0_native_k24','r0_full_k28','r1_full_k28','r1_native_k24']
    for name in order:
        r=result(); r.update(workload)
        seconds=6 if 'native' in name else 5
        r.update(median_step_seconds=seconds,global_tokens_per_second=4096/seconds)
        r['optimization'].update(dict(zip(('keep_full_layers','checkpoint_every','checkpoint_mode'),
            policies[name[3:]])))
        r['records']=[dict(step=i+1,warmup=i<3,max_rank_seconds=seconds,loss=2,
            peak_allocated_gib=39,peak_reserved_gib=42) for i in range(15)]
        path=root/name;path.mkdir();(path/'result.json').write_text(json.dumps(r))
        (root/(name+'.log')).write_text('done')
    (root/'status.log').write_text('\n'.join(n+' returncode=0' for n in order)+'\n')
    return spec


def test_reverse_order_confirmation(tmp_path):
    r=summarize(tmp_path,setup(tmp_path))
    assert r['candidate_over_two_percent_in_both_orders']
    assert r['paired_candidate_throughput_ratios']==[{'native_k24':1.2}]*2
    assert not r['candidate_convergence_proven']


def test_ce_only_requires_explicit_declaration_and_matching_labels(tmp_path):
    spec=setup(tmp_path)
    for name in ('r0_full_k28','r1_full_k28'):
        path=tmp_path/name/'result.json';r=json.loads(path.read_text())
        r['optimization']['ce_only_chunk_size']=128;path.write_text(json.dumps(r))
    with pytest.raises(ValueError):summarize(tmp_path,spec)
    spec['policies']={name:(*policy,128 if name=='full_k28' else 0)
                      for name,policy in spec['policies'].items()}
    assert summarize(tmp_path,spec)['complete']
    spec['policies']['full_k28']=(28,2,'full',64)
    with pytest.raises(ValueError):summarize(tmp_path,spec)


def test_duplicate_effective_policy_rejected(tmp_path):
    spec=setup(tmp_path)
    spec['policies']['full_k28']=(24,0,'full',0)
    with pytest.raises(ValueError):summarize(tmp_path,spec)


@pytest.mark.parametrize('fault',['order','source','seq','candidate','short','budget','allocator'])
def test_reject_invalid_evidence(tmp_path,fault):
    spec=setup(tmp_path);path=tmp_path/'r1_full_k28/result.json'
    r=json.loads(path.read_text())
    if fault=='order':(tmp_path/'status.log').write_text('wrong order')
    elif fault=='source':r['manifest']['implementation_sources']['sha256']='different'
    elif fault=='seq':r['seq']=512
    elif fault=='candidate':spec['candidate']='missing'
    elif fault=='short':spec['workload']['steps']=6
    elif fault=='budget':r['records'][5]['peak_reserved_gib']=43
    else:r['allocator']={'backend':'changed'}
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):summarize(tmp_path,spec)
