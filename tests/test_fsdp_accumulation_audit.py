import copy
import pytest
from scripts.audit_fsdp_accumulation import audit_result,compare


def sample(batch=2,accumulation=1,sync='each'):
    return dict(complete=True,backend='official',batch=batch,accumulation=accumulation,
        gradient_sync=sync,effective_global_batch=8,world_size=4,seq=512,model='8b',warmup=1,steps=2,
        precision='fp32/bf16',attention='flash',memory_budget_gib=42,
        records=[dict(step=i+1,warmup=i==0,max_rank_seconds=6.,loss=2.,peak_allocated_gib=38.,
                      peak_reserved_gib=40.) for i in range(3)],
        median_step_seconds=6.,global_tokens_per_second=8*512/6,
        optimization=dict(keep_full_layers=24,checkpoint_every=1,checkpoint_mode='full'),
        initial_hf_parity=dict(max_abs=0),manifest=dict(input_sha256='same-ordered-bytes',
            implementation_sources=dict(sha256='source'),pytorch='2.10',cuda='12.8',gpu='A6000'))


def specs():
    return dict(large=dict(batch=2,accumulation=1,gradient_sync='each',policy=(24,1,'full')),
        each=dict(batch=1,accumulation=2,gradient_sync='each',policy=(24,1,'full')),
        defer=dict(batch=1,accumulation=2,gradient_sync='defer',policy=(24,1,'full')))


def test_same_effective_batch_can_compare_three_paths():
    cases=dict(large=sample(),each=sample(1,2),defer=sample(1,2,'defer'))
    assert compare(cases,specs())['throughput_ratios']==dict(large=1.,each=1.,defer=1.)


@pytest.mark.parametrize('fault',['batch','effective','tps','inputs','source','policy','memory','sync'])
def test_reject_invalid_or_unmatched_evidence(fault):
    cases=dict(large=sample(),each=sample(1,2),defer=sample(1,2,'defer'))
    r=cases['defer']
    if fault=='batch':r['batch']=2
    elif fault=='effective':r['effective_global_batch']=4
    elif fault=='tps':r['global_tokens_per_second']/=2
    elif fault=='inputs':r['manifest']['input_sha256']='other'
    elif fault=='source':r['manifest']['implementation_sources']['sha256']='other'
    elif fault=='policy':r['optimization']['checkpoint_every']=2
    elif fault=='memory':r['records'][-1]['peak_reserved_gib']=43
    elif fault=='sync':r['gradient_sync']='each'
    with pytest.raises(ValueError):compare(cases,specs())


def test_exchange_not_silently_pooled_with_batch_study():
    r=sample();r['optimization']['exchange_reduce']='torch'
    with pytest.raises(ValueError):audit_result(r,**specs()['large'])


def test_current_all_rank_inputs_checked_without_claiming_bitwise_math():
    cases=dict(large=sample(),each=sample(1,2),defer=sample(1,2,'defer'))
    for r in cases.values():r['manifest']['input_sha256_by_rank']=['a'*64]*4
    assert compare(cases,specs())['input_identity_scope']=='all_ranks'
    cases['defer']['manifest']['input_sha256_by_rank'][2]='b'*64
    with pytest.raises(ValueError):compare(cases,specs())
    cases['defer']['manifest'].pop('input_sha256_by_rank')
    with pytest.raises(ValueError):compare(cases,specs())
