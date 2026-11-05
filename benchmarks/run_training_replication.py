"""Independent three-way full-job replication; all phases use one visible GPU.

New tag only. Compile/save/resume preflights gate ALL formal runs. Never
restart a failed job silently, shorten plateau rules, or reuse historical v1.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


BACKENDS = {'native': ('native', 'eager'), 'compile': ('native', 'compile'),
            'triton': ('triton', 'eager')}


def schedule():
    # Latin-square rotation: each backend occupies every order position once.
    rows = []
    for seed, order in zip((20261102, 20261103, 20261104),
                           (('compile', 'native', 'triton'),
                            ('native', 'triton', 'compile'),
                            ('triton', 'compile', 'native'))):
        rows.extend(dict(seq=512, seed=seed, backend=b) for b in order)
    rows.extend(dict(seq=2048, seed=20261102, backend=b)
                for b in ('triton', 'native', 'compile'))
    return rows


def write_new(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2)


def run_stage(root, name, protocol, backend, extra=()):
    if shutil.disk_usage(root).free < 50 * 1024**3:
        raise RuntimeError('Less than 50 GiB free; refusing another job')
    optimizer, execution = BACKENDS[backend]
    output = root / name
    command = [sys.executable, '-m', 'training.convergence_job', '--protocol', str(protocol),
               '--optimizer', optimizer, '--execution', execution, '--output', str(output), *extra]
    print('START', name, flush=True)
    start = time.perf_counter()
    offset = 0
    pending = ''
    # Observer lives in the parent, never creates the child's output directory.
    # Append on resume; each segment has a distinct process clock.
    with (root / f'{name}.log').open('a') as log, (root / f'{name}.clock.jsonl').open('a') as clock:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        write = lambda row: (clock.write(json.dumps(row) + '\n'), clock.flush())
        write(dict(event='spawn', pid=child.pid, command=command, epoch=time.time()))
        source = output / 'events.jsonl'
        # Resumed phase: do not retimestamp events from previous process.
        if '--resume' in extra and source.exists():
            offset = source.stat().st_size
        while True:
            code = child.poll()
            if source.exists():
                with source.open() as stream:
                    stream.seek(offset); pending += stream.read(); offset = stream.tell()
                while '\n' in pending:
                    line, pending = pending.split('\n', 1)
                    if line:
                        write(dict(event='observed', process_elapsed_seconds=time.perf_counter()-start,
                                   job_event=json.loads(line)))
            if code is not None:
                break
            time.sleep(.25)
        wall = time.perf_counter() - start
        write(dict(event='exit', returncode=code, wall_seconds=wall, pid=child.pid))
    print('END', name, code, round(wall, 2), flush=True)
    if code:
        raise RuntimeError(f'{name} failed ({code}); inspect {root / (name + ".log")}')
    return wall


def gate_preflight(root, seq):
    def losses(backend):
        rows = [json.loads(line) for line in (root / f'preflight_s{seq}_{backend}' / 'events.jsonl').read_text().splitlines()]
        return {(r['event'], r['step']): r['loss'] for r in rows if 'loss' in r}
    reference = losses('native')
    evidence = {}
    for backend in ('compile', 'triton'):
        candidate = losses(backend)
        if candidate.keys() != reference.keys():
            raise RuntimeError('Preflight did not cover identical steps/evaluations')
        differences = {f'{event}:{step}': abs(reference[(event, step)] - value)
                       for (event, step), value in candidate.items()}
        for key, difference in differences.items():
            tolerance = .002 if key.startswith('initial_validation') else (.03 if key.startswith('train') else .01)
            if not difference <= tolerance:
                raise RuntimeError(f'{backend} preflight mismatch {key}: {difference} > {tolerance}')
        evidence[backend] = differences
    write_new(root / f'preflight_s{seq}_gate.json', evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    root = Path('artifacts') / f'training_replication_{args.tag}'
    root.mkdir(exist_ok=False)
    base = json.loads(Path('training/convergence_preflight_v1.json').read_text())
    manifest = Path(base['corpus']) / 'tokens_manifest.json'
    base['data_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = dict(jobs=schedule(), min_delta=.002, patience=8, min_checks=10,
        preflight_tolerances=dict(initial_validation=.002, validation=.01, train=.03),
        max_seconds_per_job=21600, seed_scope='actual training-block permutation; fixed pretrained initialization',
        compile_scope='HF loss fullgraph=True dynamic=False; AOT backward, native fused AdamW unchanged',
        evaluation_scope='all three backends eager HF eval/test; pinned Transformers 4.51 compiled eval explicit-mask incompatibility',
        compile_cost='included, no warmup training updates; per-process first calls and recompilation included',
        compile_cache='per-job isolated Inductor/Triton cache; preflight cache not reused',
        timing='parent spawn-to-exit; all loading/training/eval/checkpoint/final test; shared preprocessing excluded',
        scheduling='one GPU sequential; S512 Latin-square backend rotation; S2048 single-seed extension',
        quality_target='per seed/sequence native minimum validation + frozen min_delta; no test-based tuning',
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in
                       ('training/convergence_job.py', 'training/triton_adamw.py',
                        'training/convergence_rules.py', __file__)})
    write_new(root / 'plan.json', plan)
    for seq in (512, 2048):
        preflight = dict(base, seq=seq, max_seconds=3600, eval_every=2, log_every=1)
        path = root / f'preflight_s{seq}.json'; write_new(path, preflight)
        for backend in ('native', 'compile', 'triton'):
            # Each stage gets its own compile cache, not one warmed by preflight.
            cache = (root / f'cache_preflight_s{seq}_{backend}').resolve()
            os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache / 'inductor')
            os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
            name = f'preflight_s{seq}_{backend}'
            run_stage(root, name, path, backend, ('--interrupt-after', '2'))
            run_stage(root, name, path, backend, ('--resume', '--interrupt-after', '4'))
        gate_preflight(root, seq)
    for job in plan['jobs']:
        name = f's{job["seq"]}_seed{job["seed"]}_{job["backend"]}'
        cache = (root / f'cache_{name}').resolve()
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache / 'inductor')
        os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
        cfg = dict(base, seq=job['seq'], seed=job['seed'], max_seconds=21600,
                   eval_every=250 if job['seq']==512 else 64, log_every=25)
        protocol = root / f'{name}.protocol.json'; write_new(protocol, cfg)
        run_stage(root, name, protocol, job['backend'])
        # Incremental report retains non-convergence/missing crossings as such.
        subprocess.run([sys.executable, '-m', 'benchmarks.summarize_training_replication',
                        '--root', str(root)], check=True)
    print('ALL_JOBS_TERMINATED', str(root), flush=True)


if __name__ == '__main__':
    main()
