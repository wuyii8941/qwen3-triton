import hashlib
import json
import pytest
from scripts.summarize_nccl_protocol import SIZES,ORDER,summarize


def setup(root):
    for name,proto in ORDER.items():
        folder=root/name;folder.mkdir();rows=[]
        for size,numel in SIZES.items():
            for op in ('all_gather','reduce_scatter'):
                ms=10. if proto=='auto' else 5.;payload=numel*(2 if op=='all_gather' else 4)
                kernel='nccl_'+op.replace('_','')+'_'+('LL' if proto=='auto' else 'SIMPLE')
                trace=folder/f'{size}_{op}_trace.json';trace.write_text(json.dumps(dict(
                    traceEvents=[dict(cat='kernel',name=kernel)])))
                rows.append(dict(size=size,operation=op,global_parameter_numel=numel,
                    dtype='torch.bfloat16' if op=='all_gather' else 'torch.float32',payload_bytes=payload,
                    timings=[dict(max_rank_cuda_ms=ms,max_rank_wall_ms=ms+1) for _ in range(20)],
                    median_cuda_ms=ms,algorithm_GB_s=payload/(ms/1000)/1e9,
                    estimated_ring_bus_GB_s=payload/(ms/1000)/1e9*.75,correctness_before_and_after=True,
                    trace_source=str(trace),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                    observed_kernels=[kernel]))
        r=dict(complete=True,world_size=4,expected_proto=proto,steps=20,warmup=5,pytorch='2.10',cuda='12.8',
            gpu='A6000',source_sha256='source',nccl=[2,27,5],
            environment=dict(NCCL_PROTO=None if proto=='auto' else 'Simple',NCCL_ALGO=None),records=rows)
        (folder/'result.json').write_text(json.dumps(r))
    (root/'status.log').write_text('\n'.join(f'{name} returncode=0' for name in ORDER)+'\n')


def test_complete_protocol_comparison(tmp_path):
    setup(tmp_path);r=summarize(tmp_path)
    assert all(r['simple_over_two_percent_in_both_orders'].values())
    assert set(r['paired_comparisons'][0]['simple_throughput_ratios'].values())=={2.}


@pytest.mark.parametrize('fault',['order','source','transport','label','payload','dtype','timings','bandwidth',
    'trace','trace_hash','kernel','coverage','correctness'])
def test_reject_unverified_or_unmatched_probe(tmp_path,fault):
    setup(tmp_path);path=tmp_path/'r1_simple/result.json';r=json.loads(path.read_text());row=r['records'][0]
    if fault=='order':(tmp_path/'status.log').write_text('incomplete')
    elif fault=='source':r['source_sha256']='changed'
    elif fault=='transport':r['environment']['NCCL_ALGO']='Tree'
    elif fault=='label':r['environment']['NCCL_PROTO']=None
    elif fault=='payload':row['payload_bytes']+=1
    elif fault=='dtype':row['dtype']='torch.float16'
    elif fault=='timings':row['timings'].pop()
    elif fault=='bandwidth':row['algorithm_GB_s']*=2
    elif fault=='trace':(tmp_path/'r1_simple/small_all_gather_trace.json').write_text('{}')
    elif fault=='trace_hash':row['trace_sha256']='wrong'
    elif fault=='kernel':row['observed_kernels']=['LL']
    elif fault=='coverage':r['records'].pop()
    else:row['correctness_before_and_after']=False
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):summarize(tmp_path)
