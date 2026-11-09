"""Audit protocol ABBA, actual kernel traces and bandwidth arithmetic."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ORDER={'r0_auto':'auto','r0_simple':'Simple','r1_simple':'Simple','r1_auto':'auto'}
SIZES={'small':1<<20,'decoder':192946432,'head':151936*4096}


def audit(result,root,expected):
    if result.get('complete') is not True or result['world_size']!=4 or result['expected_proto']!=expected:
        raise ValueError('Completed labeled four-rank probe required')
    if result['steps']<20 or result['warmup']<5:raise ValueError('Insufficient measured/warmup repetitions')
    env=dict(result['environment']);proto=env.pop('NCCL_PROTO')
    if (expected=='auto' and proto not in (None,'')) or (expected=='Simple' and proto!='Simple'):
        raise ValueError('NCCL_PROTO label mismatch')
    signature=tuple(result[k] for k in ('world_size','steps','warmup','pytorch','cuda','gpu','source_sha256'))+(
        json.dumps(result['nccl']),json.dumps(env,sort_keys=True))
    indexed={}
    for row in result['records']:
        key=(row['size'],row['operation'])
        if key in indexed or row['size'] not in SIZES or row['operation'] not in ('all_gather','reduce_scatter'):
            raise ValueError('Duplicate/unknown payload')
        numel=SIZES[row['size']];gather=row['operation']=='all_gather'
        if (row['global_parameter_numel']!=numel or row['dtype']!=('torch.bfloat16' if gather else 'torch.float32')
                or row['payload_bytes']!=numel*(2 if gather else 4) or row.get('correctness_before_and_after') is not True):
            raise ValueError('Payload dtype/size/correctness mismatch')
        if len(row['timings'])!=result['steps']:raise ValueError('Incomplete timing records')
        for timing in row['timings']:
            for field in ('max_rank_cuda_ms','max_rank_wall_ms'):
                value=timing[field]
                if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                    raise ValueError('Invalid timing')
        median=statistics.median(t['max_rank_cuda_ms'] for t in row['timings'])
        bandwidth=row['payload_bytes']/(median/1000)/1e9
        for field,value in (('median_cuda_ms',median),('algorithm_GB_s',bandwidth),
                            ('estimated_ring_bus_GB_s',bandwidth*.75)):
            if not math.isclose(row[field],value,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('Latency/bandwidth inconsistent with raw data')
        path=Path(root)/f'{key[0]}_{key[1]}_trace.json';raw=path.read_bytes()
        if Path(row['trace_source']).resolve()!=path.resolve() or hashlib.sha256(raw).hexdigest()!=row['trace_sha256']:
            raise ValueError('Trace path/hash mismatch')
        kernels=sorted({e['name'] for e in json.loads(raw)['traceEvents'] if e.get('cat')=='kernel'
            and 'nccl' in e.get('name','').lower() and key[1].replace('_','') in e.get('name','').lower()})
        if not kernels or kernels!=row['observed_kernels'] or (expected=='Simple' and
                not all('simple' in n.lower() for n in kernels)):
            raise ValueError('Unverified actual collective/protocol kernel')
        indexed[key]=row
    if set(indexed)!={(size,op) for size in SIZES for op in ('all_gather','reduce_scatter')}:
        raise ValueError('Incomplete payload coverage')
    return signature,indexed


def summarize(root):
    root=Path(root)
    if (root/'status.log').read_text().splitlines()!=[f'{name} returncode=0' for name in ORDER]:
        raise ValueError('Complete successful auto/Simple/Simple/auto order required')
    signatures=set();records=[];indexed={}
    for name,proto in ORDER.items():
        path=root/name/'result.json';raw=path.read_bytes();r=json.loads(raw)
        signature,payloads=audit(r,root/name,proto);signatures.add(signature);indexed[name]=payloads
        records.append(dict(case=name,source=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            environment=r['environment'],records=r['records']))
    if len(signatures)!=1:raise ValueError('Different implementation/runtime/transport settings across probes')
    comparisons=[]
    for repeat in (0,1):
        ratios={f'{size}/{op}':indexed[f'r{repeat}_auto'][size,op]['median_cuda_ms']/
            indexed[f'r{repeat}_simple'][size,op]['median_cuda_ms'] for size in SIZES for op in ('all_gather','reduce_scatter')}
        comparisons.append(dict(repeat=repeat,simple_throughput_ratios=ratios))
    stable={key:all(c['simple_throughput_ratios'][key]>1.02 for c in comparisons)
            for key in comparisons[0]['simple_throughput_ratios']}
    return dict(complete=True,records=records,paired_comparisons=comparisons,
        simple_over_two_percent_in_both_orders=stable,
        scope='Isolated collective diagnostic; neither full-model speed nor convergence established; defaults unchanged')


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=summarize(a.root)
    with a.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report['paired_comparisons'],indent=2))


if __name__=='__main__':main()
