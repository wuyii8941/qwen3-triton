import pytest
import torch
from benchmarks.nccl_protocol_probe import layer_parameter_count,make_buffers,check_output,profile_collective


def test_real_block_parameter_count():
    assert layer_parameter_count()==192946432


@pytest.mark.parametrize('operation',['all_gather','reduce_scatter'])
def test_payload_checks_rank_order_boundaries_and_middle(operation):
    inputs=[make_buffers(32,4,r,operation,'cpu')[0] for r in range(4)]
    for rank in range(4):
        actual=torch.cat(inputs) if operation=='all_gather' else torch.stack(inputs).sum(0).chunk(4)[rank].clone()
        assert check_output(actual,4,rank,operation)
        actual[2]=99
        assert not check_output(actual,4,rank,operation)


def test_reject_invalid_payload():
    for shape in (1,7,15):
        with pytest.raises(ValueError):make_buffers(shape,4,0,'all_gather','cpu')
    with pytest.raises(ValueError):make_buffers(32,4,0,'unknown','cpu')


def test_profiler_boundaries_use_cpu_barriers_before_next_gpu_work():
    events=[]
    class Profiler:
        def __enter__(self):events.append('enter')
        def __exit__(self,*args):events.append('exit')
    profile_collective(lambda:events.append('collective'),lambda:events.append('cuda_sync'),
        Profiler(),lambda:events.append('cpu_barrier'))
    assert events==['enter','cpu_barrier','collective','cuda_sync','cpu_barrier','exit','cpu_barrier']
