"""Finite, resumable correctness/performance matrix, isolated child per run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='artifacts/optimization_matrix_v1')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(exist_ok=args.resume)
    report = dict(start_time=time.time(), jobs=[], scope='Short fixed-batch screening; no convergence or checkpoints.',
                  gpu_environment=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  versions={name:importlib.metadata.version(name) for name in ('torch','triton','transformers','flashoptim')},
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      map(Path, ['benchmarks/check_tied_embedding.py','benchmarks/benchmark_tied_embedding_step.py',
                                 'training/multi_tensor_flashoptim.py'])})
    def save(): (root/'progress.json').write_text(json.dumps(report,indent=2))
    def run(name, command, output):
        if args.resume and output.exists():
            previous = json.loads(output.read_text())
            if previous.get('passed', previous.get('median_step_ms') is not None):
                report['jobs'].append(dict(name=name,status='reused',output=str(output)))
                save();return True
        start = time.time()
        report['active'] = name;save()
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with (root/f'{name}.log').open('w') as stream:
            result = subprocess.run([sys.executable,'-m',*command], stdout=stream,stderr=subprocess.STDOUT,env=env)
        ok = result.returncode == 0 and output.exists()
        if ok and 'gate' in name: ok = bool(json.loads(output.read_text()).get('passed'))
        report['jobs'].append(dict(name=name,status='passed' if ok else 'failed',
                                  seconds=time.time()-start,output=str(output),returncode=result.returncode))
        report.pop('active',None);save()
        print('JOB', name, 'PASS' if ok else 'FAIL', flush=True)
        return ok
    accepted = {}
    for batch, seq in ((1,512),(1,128),(2,512),(1,2048)):
        shape = f'b{batch}_s{seq}'
        gate = Path('artifacts/tied_embedding_deterministic_v1/result.json') if (batch,seq)==(1,512) else Path(f'artifacts/tied_embedding_gate_{shape}_v1/result.json')
        if not gate.exists():
            tag = f'optimization_matrix_gate_{shape}_v1'
            gate = Path('artifacts')/tag/'result.json'
            ok = run('gate_'+shape,['benchmarks.check_tied_embedding','--tag',tag,'--seq',str(seq),
                                   '--batch',str(batch),'--deterministic','--gate-only'],gate)
        else:
            g = json.loads(gate.read_text())
            ok = bool(g.get('passed')) and g.get('deterministic') and g.get('seq')==seq and g.get('batch',1)==batch
        accepted[shape] = (ok,gate)
        variants = ('baseline','candidate','multi','both') if ok else ('baseline','multi')
        for variant in variants:
            name = f'r0_{shape}_{variant}'
            output = root/f'{name}.json'
            # Multi-only does not change embedding and can use the original gate.
            effective_gate = gate if ok else Path('artifacts/tied_embedding_deterministic_v1/result.json')
            command = ['benchmarks.benchmark_tied_embedding_step','--single-stage',variant,'--output',str(output),
                       '--seq',str(seq),'--batch',str(batch),'--steps','20','--gate',str(effective_gate)]
            if (batch,seq)==(1,512) and variant in ('baseline','both'): command += ['--profile']
            run(name,command,output)
    # Reverse order confirms promising combinations without selecting the fastest single run.
    for batch,seq in ((1,512),(1,2048)):
        shape=f'b{batch}_s{seq}';ok,gate=accepted[shape]
        if not ok: continue
        for variant in ('both','multi','candidate','baseline'):
            name=f'r1_{shape}_{variant}';output=root/f'{name}.json'
            run(name,['benchmarks.benchmark_tied_embedding_step','--single-stage',variant,'--output',str(output),
                      '--seq',str(seq),'--batch',str(batch),'--steps','20','--gate',str(gate)],output)
    report.update(complete=True, end_time=time.time());save()
    print('MATRIX_COMPLETE',str(root),flush=True)


if __name__=='__main__': main()
