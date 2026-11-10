import json
import pytest
from scripts.summarize_fsdp_confirmation import ORDER,POLICIES,profile_summary,summarize
from tests.test_fsdp_memory_summary import result


def trace():
    return dict(traceEvents=[dict(cat='kernel',name='nccl_AllGather',ts=0,dur=10),
        dict(cat='kernel',name='nccl_AllGather',ts=20,dur=20),
        dict(cat='kernel',name='gemm',ts=5,dur=20),
        dict(cat='cuda_runtime',name='cudaMalloc',ts=0,dur=1),
        dict(cat='cuda_runtime',name='cudaFree',ts=0,dur=1)])


def test_profile_uses_union_and_overlap_not_summed_work():
    r=profile_summary(trace())
    assert r['summed_kernel_work_ms']==.05
    assert r['comm_union_ms']==.03 and r['local_union_ms']==.02
    assert r['comm_local_overlap_ms']==.01 and r['gpu_kernel_union_ms']==.04
    assert r['cuda_malloc_calls']==r['cuda_free_calls']==1
    assert r['all_gather_kernel_count']==2
    assert r['allocation_api_summed_duration_ms']==.002
    assert r['training_scoped_allocation_calls'] is None


def test_allocation_scope_keeps_api_waits_separate_from_gpu_work():
    t=trace();t['traceEvents'].append(dict(cat='user_annotation',name='comparison.forward',ts=0,dur=10))
    r=profile_summary(t)
    assert r['training_scoped_allocation_calls']==2
    assert r['training_scoped_allocation_summed_duration_ms']==.002
    assert r['gpu_kernel_union_ms']==.04


def write_round(root):
    for name in ORDER:
        policy=name[3:];r=result();seconds={'native_k24':6.,'native_k30':5.8,'full_k35':5.5}[policy]
        r.update(steps=12,precision='FP32 master/AdamW/reduce, BF16 compute',attention='forced Flash SDPA',
            median_step_seconds=seconds,global_tokens_per_second=2048/seconds)
        r['records']=[dict(step=i+1,warmup=i<3,max_rank_seconds=seconds,loss=2.,
            peak_allocated_gib=39.,peak_reserved_gib=42.) for i in range(15)]
        r['optimization'].update(dict(zip(('keep_full_layers','checkpoint_every','checkpoint_mode'),POLICIES[policy])))
        path=root/name;path.mkdir();(path/'result.json').write_text(json.dumps(r))
        (root/(name+'.log')).write_text('completed')
        if name.startswith('r0_'):(path/'trace.json').write_text(json.dumps(trace()))
    (root/'status.log').write_text('\n'.join(n+' returncode=0' for n in ORDER)+'\n')


def test_confirmation_preserves_paired_orders_and_raw_timings(tmp_path):
    write_round(tmp_path);r=summarize(tmp_path)
    assert r['complete'] and r['candidate_over_two_percent_in_both_orders']
    assert not r['candidate_convergence_proven'] and len(r['paired_comparisons'])==2
    assert all(len(x['steady_step_seconds'])==12 for x in r['records'])


@pytest.mark.parametrize('fault',['order','incomplete','source','steps','precision','missing_trace'])
def test_reject_invalid_confirmation(tmp_path,fault):
    write_round(tmp_path);path=tmp_path/'r1_full_k35/result.json';r=json.loads(path.read_text())
    if fault=='order':(tmp_path/'status.log').write_text('r0_native_k24 returncode=0\n')
    elif fault=='incomplete':r['complete']=False
    elif fault=='source':r['manifest']['implementation_sources']['sha256']='different'
    elif fault=='steps':r['steps']=6
    elif fault=='precision':r['precision']='different'
    else:(tmp_path/'r0_full_k35/trace.json').unlink()
    path.write_text(json.dumps(r))
    with pytest.raises((ValueError,FileNotFoundError)):summarize(tmp_path)
