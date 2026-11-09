"""Audit all-rank replay records; does not independently replay training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def audit(root):
    root=Path(root)
    summary=json.loads((root/'result.json').read_text())
    if summary.get('passed') is not True or summary.get('world_size')!=4 or summary.get('replay_steps')!=2:
        raise ValueError('Successful four-rank two-step replay required')
    tiny=summary.get('synthetic_tiny')
    if type(tiny) is not bool or summary.get('full_8b_resume_proven') is not (not tiny):
        raise ValueError('Tiny/full-8B evidence scope mismatch')
    records=[];sources=set();counts=set();schemas=[]
    def digest(value):
        if not isinstance(value,dict) or not re.fullmatch('[0-9a-f]{64}',value.get('sha256','')):
            raise ValueError('Missing tensor hash')
        if not isinstance(value.get('shape'),list) or not isinstance(value.get('local_shape'),list):
            raise ValueError('Missing tensor shape')
    for rank in range(4):
        path=root/f'rank{rank}_result.json';raw=path.read_bytes();r=json.loads(raw)
        if (r.get('passed') is not True or r.get('rank')!=rank or r.get('world_size')!=4
                or r.get('deterministic') is not True):
            raise ValueError('Incomplete or mislabeled rank record')
        sources.add(r['source_sha256'])
        for name in ('initial_updates','replay_updates'):
            if len(r[name])!=2:raise ValueError('Two initial/replayed updates required')
            for step in r[name]:
                if not math.isfinite(step['loss']) or not re.fullmatch('[0-9a-f]{64}',step['input_sha256']):
                    raise ValueError('Invalid loss or input digest')
        state=r['state'];parameters=state['parameters'];optim=state['optimizer_states']
        if not parameters or set(parameters)!=set(optim):raise ValueError('Optimizer parameter coverage mismatch')
        count=0
        for name,value in parameters.items():
            digest(value)
            if value['dtype']!='torch.float32':raise ValueError('FP32 master required')
            count+=math.prod(value['shape'])
            for moment in ('exp_avg','exp_avg_sq'):
                m=optim[name][moment];digest(m)
                if any(m[k]!=value[k] for k in ('dtype','shape','local_shape')):
                    raise ValueError('Optimizer moment schema mismatch')
            digest(optim[name]['step'])
        if count!=r['resources']['parameters']:raise ValueError('Parameter count differs from resource plan')
        if not tiny and count!=8190735360:raise ValueError('Not the declared real 8B model')
        if summary.get('wrote_checkpoint') is False:
            if (r.get('wrote_checkpoint') is not False or r.get('reader')!=summary.get('reader')
                    or not r['resources'].get('read_only_replay_resources_sufficient')):
                raise ValueError('Read-only replay guard/reader identity mismatch')
        elif not r['resources']['resources_sufficient']:raise ValueError('Resource guard did not pass')
        if 'expected_updates' in r and r['expected_updates']!=r['replay_updates']:
            raise ValueError('Reported expected/replayed update mismatch')
        if set(r['rng'])!={'torch_cpu','torch_cuda','numpy','python'}:raise ValueError('RNG stream coverage mismatch')
        for values in r['rng'].values():
            if len(values)!=3 or not all(math.isfinite(x) for x in values):raise ValueError('Invalid RNG samples')
        schemas.append({k:(v['shape'],v['dtype']) for k,v in parameters.items()})
        counts.add(count)
        records.append(dict(rank=rank,source=str(path),sha256=hashlib.sha256(raw).hexdigest()))
    if len(sources)!=1 or len(counts)!=1 or any(s!=schemas[0] for s in schemas):
        raise ValueError('Cross-rank source/parameter schema mismatch')
    return dict(passed=True,synthetic_tiny=tiny,full_8b_resume_proven=not tiny,
        parameters=next(iter(counts)),records=records,
        scope='All-rank replay-record consistency audit; equality asserted by replay worker, not independently rerun here')


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.root)
    with a.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))


if __name__=='__main__':main()
