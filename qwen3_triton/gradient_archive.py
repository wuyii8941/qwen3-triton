"""Bounded, lossless streaming archives for real8B numerical replay."""
from pathlib import Path
import gzip,hashlib
import torch

class _CappedWriter:
    def __init__(self,handle,limit):self.handle=handle;self.limit=limit;self.bytes_written=0
    def write(self,data):
        if self.bytes_written+len(data)>self.limit:raise OSError('Compressed gradient archive exceeds declared per-rank storage cap')
        n=self.handle.write(data);self.bytes_written+=n;return n
    def flush(self):return self.handle.flush()
    def tell(self):return self.handle.tell()
    def fileno(self):return self.handle.fileno()

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda:handle.read(8*2**20),b''):digest.update(data)
    return digest.hexdigest()

def save(raw,path,*,max_bytes=6*2**30):
    with Path(path).open('xb') as handle:
        bounded=_CappedWriter(handle,max_bytes)
        with gzip.GzipFile(fileobj=bounded,mode='wb',compresslevel=1,mtime=0) as compressed:
            # Legacy serialization streams storages sequentially. Zip-format
            # random seeking inside a gzip stream would repeatedly decompress.
            torch.save(raw,compressed,_use_new_zipfile_serialization=False)
    assert Path(path).stat().st_size<=max_bytes
    return dict(sha256=sha(path),bytes=Path(path).stat().st_size,lossless=True,
        format='gzip_level1_legacy_torch_storage_stream',cap_bytes=max_bytes)

def load(path):
    with gzip.open(path,'rb') as handle:
        return torch.load(handle,map_location='cpu',weights_only=True)
