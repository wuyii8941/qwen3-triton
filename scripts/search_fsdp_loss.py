"""Serial successor to retention search; never runs alongside its predecessor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from training.fsdp_loss_search import summarize_loss_screen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait-pid', type=int, required=True)
    p.add_argument('--wait-status', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(complete=False, pid=os.getpid(), screen=[], confirmation=[])

    def save(phase, **extra):
        state.update(phase=phase, updated=time.time(), **extra)
        temporary = a.output/'status.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        os.replace(temporary, a.output/'status.json')
        print(json.dumps(dict(phase=phase, **extra)), flush=True)

    def idle():
        save('waiting_for_idle_gpus')
        while True:
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            readings = [tuple(int(x.strip()) for x in r.split(',')) for r in rows]
            if len(readings) >= 4 and all(m < 2048 and u < 10 for m, u in readings[:4]):
                return
            time.sleep(15)

    def launch(name, arguments):
        idle(); save('running_case', case=name)
        log = a.output/(name+'.log')
        with log.open('x') as stream:
            job = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=4', *arguments], stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        raw = log.read_bytes()
        return job.returncode, raw, hashlib.sha256(raw).hexdigest()

    try:
        save('waiting_for_predecessor', predecessor=a.wait_pid)
        proc = Path(f'/proc/{a.wait_pid}/cmdline')
        while proc.exists():
            try:
                command = proc.read_bytes()
            except FileNotFoundError:
                break
            if not command:
                break
            if b'search_fsdp_retention.py' not in command or str(a.wait_status.parent).encode() not in command:
                break
            time.sleep(15)
        previous = json.loads(a.wait_status.read_text())
        if not previous.get('complete') or previous.get('phase') != 'finished':
            raise RuntimeError('Retention predecessor did not finish; do not start GPU work')
        keep = previous['confirmed_keep_full_layers'] if previous.get('consistent_over_two_percent') else 0
        state.update(model=previous['model'], keep_full_layers=keep,
            budget_gib=previous['budget_gib'], seq=previous['seq'], batch=previous['batch'])
        gate = ['benchmarks/verify_block_fsdp.py', '--backend', 'official', '--bf16',
                '--head-dim', '128', '--loss-chunk-size', '7', '--keep-full-layers', '1']
        code, raw, sha = launch('gpu_loss_gate', gate)
        state['gate'] = dict(returncode=code, log_sha256=sha)
        passed = [json.loads(line) for line in raw.decode(errors='replace').splitlines()
                  if line.startswith('{"passed":')]
        if code or len(passed) != 2 or {r['tied'] for r in passed} != {False, True} or not all(
                r['passed'] and r['world'] == 4 and r['steps'] == 3 for r in passed):
            raise RuntimeError('CUDA FSDP loss gate failed; native training remains unchanged')
        save('gpu_loss_gate_passed', gate_results=passed)

        def run(chunk, name, steps):
            output = a.output/name
            args = ['benchmarks/compare_fsdp_training.py', '--backend', 'official',
                '--model', state['model'], '--keep-full-layers', str(keep),
                '--memory-budget-gib', str(state['budget_gib']), '--seq', str(state['seq']),
                '--batch', str(state['batch']), '--warmup', '3', '--steps', str(steps),
                '--loss-chunk-size', str(chunk), '--reference-check', '--output', str(output)]
            code, raw, sha = launch(name, args)
            row = dict(chunk_size=chunk, returncode=code, log_sha256=sha, source=str(output/'result.json'))
            if code:
                if b'out of memory' in raw.lower() or b'OutOfMemoryError' in raw:
                    return dict(row, status='oom')
                raise RuntimeError(f'Non-capacity failure in {name}')
            result_bytes = (output/'result.json').read_bytes()
            return dict(row, status='passed', result=json.loads(result_bytes),
                        result_sha256=hashlib.sha256(result_bytes).hexdigest())

        for chunk in (0, 32, 128, 512):
            state['screen'].append(run(chunk, f'screen_c{chunk}', 6))
            save('screen_case_finished')
            if chunk == 0 and state['screen'][-1]['status'] != 'passed':
                raise RuntimeError('Native-loss baseline does not fit budget')
        selected = summarize_loss_screen(state['screen'])
        save('screen_complete', selection=selected)
        best = selected['best_chunk_size']
        pairs = []
        if best:
            for repeat, order in enumerate(([best, 0], [0, best])):
                rows = [run(chunk, f'confirm_r{repeat}_c{chunk}', 12) for chunk in order]
                state['confirmation'].extend(rows)
                pair = summarize_loss_screen(rows)
                native = next(x['result']['median_step_seconds'] for x in rows if x['chunk_size'] == 0)
                candidate = next(x['result']['median_step_seconds'] for x in rows if x['chunk_size'] == best)
                pairs.append(dict(repeat=repeat, throughput_speedup=native/candidate))
                save('confirmation_pair_finished', paired_comparisons=pairs)
        save('finished', complete=True, confirmed_chunk_size=best,
             consistent_over_two_percent=bool(pairs) and all(x['throughput_speedup'] > 1.02 for x in pairs))
    except Exception as error:
        save('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
