import pytest
from benchmarks.benchmark_bf16_exchange import MODES,SIZES,INPUT_CONTRACT,comparisons,audit_report,source_hashes


def rows():
    return [dict(size='decoder',repeat=r,mode=mode,median_cuda_ms=2. if mode=='native_fp32' else 1.)
            for r in (0,1) for mode in MODES]


def test_candidate_requires_both_orders():
    data=rows();assert comparisons(data,['decoder'])['triton_over_two_percent_in_both_orders']['decoder']
    next(r for r in data if r['repeat']==1 and r['mode']=='exchange_triton')['median_cuda_ms']=3.
    assert not comparisons(data,['decoder'])['triton_over_two_percent_in_both_orders']['decoder']


def test_incomplete_duplicate_or_nonfinite_is_not_speedup():
    for data in (rows()[:-1],rows()+rows()[:1]):
        with pytest.raises(ValueError):comparisons(data,['decoder'])
    data=rows();data[0]['median_cuda_ms']=float('nan')
    with pytest.raises(ValueError):comparisons(data,['decoder'])


def full_report():
    n=SIZES['decoder'];data=[]
    for repeat,order in ((0,MODES),(1,tuple(reversed(MODES)))):
        for mode in order:
            latency=2. if mode=='native_fp32' else 1.
            data.append(dict(size='decoder',repeat=repeat,mode=mode,numel=n,
                median_cuda_ms=latency,median_wall_ms=latency,
                timings=[dict(max_rank_cuda_ms=latency,max_rank_wall_ms=latency)]*12,
                workspace_tensor_bytes=n*(1 if mode=='native_fp32' else 9 if mode=='exchange_torch' else 5),
                peak_extra_allocated_bytes=n*10,correctness_before_and_after=True,
                wire_element_dtype='torch.float32' if mode=='native_fp32' else 'torch.bfloat16',
                wire_representation_bytes=n*(4 if mode=='native_fp32' else 2),network_bytes_measured=False))
    report=dict(complete=True,world_size=4,expected_proto='auto',steps=12,warmup=3,
        source_hashes=source_hashes(),selected_sizes=['decoder'],environment=dict(NCCL_PROTO=None),
        input_contract=INPUT_CONTRACT,protocol_verified_by_trace=False,
        input_prefix_samples=dict(decoder=['a'*64]*4),records=data)
    report.update(comparisons(data,['decoder']));return report


def test_full_audit_recomputes_raw_decision():
    assert audit_report(full_report(),'auto')['triton_over_two_percent_in_both_orders']['decoder']


@pytest.mark.parametrize('fault',['median','raw','source','proto','workspace','decision','order','contract','wire'])
def test_audit_rejects_mislabeled_or_incomplete_evidence(fault):
    r=full_report()
    if fault=='median':r['records'][0]['median_cuda_ms']=1.
    elif fault=='raw':r['records'][0]['timings']=[]
    elif fault=='source':r['source_hashes']={}
    elif fault=='proto':r['environment']['NCCL_PROTO']='Simple'
    elif fault=='workspace':r['records'][0]['workspace_tensor_bytes']=0
    elif fault=='decision':r['triton_over_two_percent_in_both_orders']['decoder']=False
    elif fault=='order':r['records'].reverse()
    elif fault=='contract':r['input_contract']='arbitrary FP32 quantization'
    elif fault=='wire':r['records'][0]['wire_element_dtype']='torch.bfloat16'
    with pytest.raises(ValueError):audit_report(r,'auto')
