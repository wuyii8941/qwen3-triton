from types import SimpleNamespace
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.planner import LoadPlan
from benchmarks.dcp_sorted_reader import ordered_plan,OffsetSortedFileSystemReader


def test_sort_keeps_items_and_plan_metadata_without_mutation():
    items=[SimpleNamespace(storage_index=i) for i in range(4)]
    metadata={i:SimpleNamespace(relative_path=f'{i%2}.distcp',offset=100-i*10) for i in range(4)}
    marker=object();plan=LoadPlan(items=items,planner_data=marker,storage_data=marker)
    result=ordered_plan(plan,metadata)
    assert plan.items==items
    assert result.planner_data is marker and result.storage_data is marker
    assert [r.storage_index for r in result.items]==[2,0,3,1]
    assert {id(x) for x in result.items}=={id(x) for x in items}


def test_bad_storage_offset_rejected():
    plan=LoadPlan(items=[SimpleNamespace(storage_index=0)])
    with pytest.raises(ValueError):ordered_plan(plan,{0:SimpleNamespace(relative_path='a',offset=-1)})


def test_actual_cpu_dcp_roundtrip_preserves_tensors_and_byte_metadata(tmp_path):
    source=dict(small=torch.arange(3,dtype=torch.float32),
        weight=torch.arange(512,dtype=torch.float32).reshape(32,16),
        exp_avg=torch.full((32,16),.125),exp_avg_sq=torch.full((32,16),.25),
        metadata=dict(step=2,tag='roundtrip'))
    dcp.save(source,checkpoint_id=tmp_path/'state')
    destination={k:torch.zeros_like(v) if isinstance(v,torch.Tensor) else dict(step=0,tag='')
                 for k,v in source.items()}
    class RecordingReader(OffsetSortedFileSystemReader):
        def __init__(self,path):super().__init__(path);self.observed=[]
        def _slice_file(self,file,sinfo):
            self.observed.append((sinfo.relative_path,sinfo.offset))
            return super()._slice_file(file,sinfo)
    reader=RecordingReader(tmp_path/'state')
    dcp.load(destination,storage_reader=reader)
    assert len(reader.observed)>1 and reader.observed==sorted(reader.observed)
    for key,value in source.items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(destination[key],value,rtol=0,atol=0)
        else:assert destination[key]==value
