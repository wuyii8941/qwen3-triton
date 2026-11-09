"""Independent all-rank coverage/finite/full-byte-fingerprint audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def check_tensor(record,*,shape=None,local_shape=None):
    if (record.get('finite') is not True or record.get('dtype')!='torch.float32' or
            not re.fullmatch('[0-9a-f]{64}',record.get('sha256',''))):
        raise ValueError('Finite FP32 complete byte hash required')
    for key in ('shape','local_shape'):
        if not isinstance(record.get(key),list) or any(type(v) is not int or v<1 for v in record[key]):
            raise ValueError('Valid tensor geometry required')
    if shape is not None and record['shape']!=shape:raise ValueError('Wrong global shape')
    if local_shape is not None and record['local_shape']!=local_shape:raise ValueError('Wrong local shape')


def audit(root):
    raw_summary=(root/'result.json').read_bytes();summary=json.loads(raw_summary)
    if (summary.get('passed') is not True or summary['world']!=2 or summary['steps']!=3 or
            summary['real_parameters']!=1720574976 or summary['parameter_entries_per_rank']!=310):
        raise ValueError('Complete real1.7B two-rank gate required')
    if len(summary['raw_reports'])!=2 or [r['rank'] for r in summary['raw_reports']]!=[0,1]:
        raise ValueError('All rank reports required exactly once')
    reports=[];source_snapshots=set()
    phases=('native_a','native_b','keep28','checkpoint28')
    for rank,provenance in enumerate(summary['raw_reports']):
        path=root/'gate'/f'rank{rank}_result.json';raw=path.read_bytes();report=json.loads(raw)
        if Path(provenance['path']).resolve()!=path.resolve() or hashlib.sha256(raw).hexdigest()!=provenance['sha256']:
            raise ValueError('Changed rank evidence')
        if (report['rank']!=rank or report['world']!=2 or report.get('passed') is not True or
                report['deterministic'] is not True or report['steps_per_phase']!=3 or
                report['real_parameters']!=1720574976 or report['parameter_entries_per_rank']!=310 or
                report['worker_sha256']!=summary['worker_sha256'] or set(report['phases'])!=set(phases)):
            raise ValueError('Rank contract mismatch')
        snapshot=report['implementation_sources']
        encoded=json.dumps(snapshot['files'],sort_keys=True,separators=(',',':')).encode()
        if hashlib.sha256(encoded).hexdigest()!=snapshot['sha256']:raise ValueError('Invalid source snapshot')
        source_snapshots.add(json.dumps(snapshot,sort_keys=True))
        base=report['phases']['native_a']
        if len(base)!=3 or any(report['phases'][p]!=base for p in phases):
            raise ValueError('Full candidate/reference records not exactly equal')
        names=None
        for step,record in enumerate(base,1):
            params=record['parameters'];grads=record['gradients'];states=record['optimizer_states']
            if (record['step']!=step or not math.isfinite(record['loss']) or
                    not re.fullmatch('[0-9a-f]{64}',record['input_sha256']) or len(params)!=310 or
                    set(grads)!=set(params) or set(states)!=set(params)):
                raise ValueError('Incomplete parameter/gradient/optimizer coverage')
            if names is not None and set(params)!=names:raise ValueError('Parameter coverage changed')
            names=set(params)
            if sum(math.prod(p['shape']) for p in params.values())!=1720574976:
                raise ValueError('Global element coverage mismatch')
            for name,param in params.items():
                shape=param['shape']
                if not shape or shape[0]%2:raise ValueError('Expected evenly split dim0 shards')
                local=[shape[0]//2]+shape[1:]
                check_tensor(param,shape=shape,local_shape=local)
                check_tensor(grads[name],shape=shape,local_shape=local)
                state=states[name]
                if set(state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('All AdamW state entries required')
                for key in ('exp_avg','exp_avg_sq'):check_tensor(state[key],shape=shape,local_shape=local)
                check_tensor(state['step'],shape=[],local_shape=[])
                if state['step']['sha256']!=hashlib.sha256(struct.pack('<f',float(step))).hexdigest():
                    raise ValueError('Optimizer step counter does not match update')
            for buffer in record['buffers'].values():check_tensor(buffer)
            groups=record['param_groups']
            if (len(groups)!=1 or groups[0]['lr']!=1e-5 or groups[0]['betas']!=[.9,.999] or
                    groups[0]['eps']!=1e-8 or groups[0]['weight_decay']!=.01 or groups[0]['fused'] is not True):
                raise ValueError('Unexpected AdamW hyperparameters')
        reports.append(report)
    if len(source_snapshots)!=1:raise ValueError('Ranks used different sources')
    for step in range(3):
        a,b=(r['phases']['native_a'][step] for r in reports)
        if set(a['parameters'])!=set(b['parameters']):raise ValueError('Ranks cover different parameter sets')
        for name in a['parameters']:
            pa,pb=a['parameters'][name],b['parameters'][name]
            if pa['shape']!=pb['shape'] or math.prod(pa['local_shape'])+math.prod(pb['local_shape'])!=math.prod(pa['shape']):
                raise ValueError('Rank shards do not cover global parameters')
    log=(root/'worker.log').read_bytes()
    if hashlib.sha256(log).hexdigest()!=summary['log_sha256']:raise ValueError('Changed worker log')
    worker=Path('benchmarks/verify_fsdp_1p7b_states.py')
    return dict(passed=True,world=2,real_parameters=1720574976,parameter_entries_per_rank=310,
        steps=3,candidate_phases=list(phases[1:]),full_gradients_exact=True,
        full_fp32_masters_exact=True,full_adamw_states_exact=True,buffers_and_groups_exact=True,
        raw_summary_path=str((root/'result.json').resolve()),
        raw_summary_sha256=hashlib.sha256(raw_summary).hexdigest(),raw_reports=summary['raw_reports'],
        implementation_sha256=reports[0]['implementation_sources']['sha256'],
        current_worker_hash_matches=hashlib.sha256(worker.read_bytes()).hexdigest()==summary['worker_sha256'],
        frozen_content_sha256=summary['frozen_content_sha256'],defaults_changed=False,convergence_proven=False,
        scope='All-rank record/coverage/finite/byte-hash audit for three deterministic real1.7B updates; not independent GPU replay, long quality, performance or changed-world proof')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.root)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
