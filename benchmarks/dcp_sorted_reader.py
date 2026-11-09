"""Opt-in PyTorch 2.10 filesystem-reader experiment; not a training change.

Only reorder independent ReadItems by file and physical offset. Tensor slicing,
deserialization, destination resolution and commit remain PyTorch's code.
Never monkeypatch the native reader or alter an in-flight checkpoint load.
"""
from dataclasses import replace

from torch.distributed.checkpoint import FileSystemReader


def ordered_plan(plan, storage_data):
    def location(item):
        record=storage_data[item.storage_index]
        if not isinstance(record.relative_path,str) or type(record.offset) is not int or record.offset<0:
            raise ValueError('Expected PyTorch filesystem storage path and nonnegative offset')
        return record.relative_path,record.offset
    return replace(plan,items=sorted(plan.items,key=location))


class OffsetSortedFileSystemReader(FileSystemReader):
    def read_data(self,plan,planner):
        return super().read_data(ordered_plan(plan,self.storage_data),planner)
