"""Lossless raw profiler storage; decoding never changes or filters events."""
import gzip,hashlib
from collections import Counter
from pathlib import Path


class TraceSource:
    """Byte-compatible input for historical geometry auditors, including gzip."""
    def __init__(self,path):
        self.path=Path(path)
        self.stored=self.path.read_bytes()
        self.raw=gzip.decompress(self.stored) if self.path.suffix=='.gz' else self.stored

    def read_bytes(self):return self.raw

    def read_text(self):return self.raw.decode('utf-8')

    @property
    def artifact_sha256(self):return hashlib.sha256(self.stored).hexdigest()

    @property
    def decoded_sha256(self):return hashlib.sha256(self.raw).hexdigest()


def locate_trace(folder,name):
    plain=Path(folder)/name;compressed=plain.with_suffix(plain.suffix+'.gz')
    if plain.exists() and compressed.exists():raise ValueError('Ambiguous raw trace: both plain and gzip artifacts exist')
    if compressed.exists():return compressed
    if plain.exists():return plain
    raise FileNotFoundError(plain)


def validation_geometry(events,plan):
    """Original batch4/rank: local2048 tokens, vocabulary projection8192."""
    table={('gemm',e['M'],e['N'],e['K'],*(x for s in e['strides'] for x in s)):e for e in plan['entries'] if e['kind']=='gemm'}
    expected=Counter()
    def add(m,n,k,am,ak,bk,bn,count):
        entry=table.get(('gemm',m,n,k,am,ak,bk,bn))
        if entry:assert entry['family']=='tiled' and entry['splits']==1
        tm,tn,tk,group,warps,stages=entry['config'] if entry else (64,64,32,8,4,3)
        expected[((((m+tm-1)//tm)*((n+tn-1)//tn),1,1),(warps*32,1,1))]+=count
    for out,inp,count in [(4096,4096,72),(1024,4096,72),(4096,12288,36)]:
        add(2048,out,inp,inp,1,1,inp,count*32)
    add(8192,37984,4096,4096,1,1,4096,32)
    assert sum(expected.values())==181*32
    actual=Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name']=='_gemm')
    entry=next((e for e in plan['entries'] if e['kind']=='gate_up' and (e['M'],e['N'],e['K'])==(2048,12288,4096)),None)
    tm,tn,tk,group,warps,stages=entry['config'] if entry else (64,64,32,8,4,3)
    gate_expected=Counter({((((2048+tm-1)//tm)*((12288+tn-1)//tn),1,1),(warps*32,1,1)):36*32})
    gate_actual=Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name']=='_gate_up')
    return expected==actual and gate_expected==gate_actual
