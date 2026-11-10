"""Three paired process rounds, unchanged data/precision, warmed on-disk JIT cache.

Not a convergence experiment. Record startup + JIT, shared GPU load and every
validation point; withhold acceptance on >.01 validation loss deviation.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def gpu():
    return subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args()
    root=Path('artifacts')/f'flashoptim_confirmation_{a.tag}';root.mkdir(exist_ok=False)
    plan=dict(rounds=3,steps=64,lr=1e-5,seq=512,val_abs_tolerance=.01,
        orders=[['native','flashoptim'],['flashoptim','native'],['native','flashoptim']],
        precision='BF16/no master in both; no gradient release; same seed/data/input/validation',
        cache='Reuses disk JIT cache populated by initial screen; first/cold experiment reported separately.',
        scope='Short validation stability and matched-config step confirmation; NOT convergence, no test-based tuning.')
    (root/'plan.json').write_text(json.dumps(plan,indent=2));rounds=[]
    for index,order in enumerate(plan['orders']):
        result={}
        for variant in order:
            name=f'r{index}_{variant}';path=root/f'{name}.json';before=gpu();tick=time.perf_counter()
            print('START',name,flush=True)
            with (root/f'{name}.log').open('x') as out:
                code=subprocess.run([sys.executable,'-m','benchmarks.screen_research_candidates',
                    '--mode','train','--variant',variant,'--lr','1e-5','--steps','64','--output',str(path)],
                    stdout=out,stderr=subprocess.STDOUT).returncode
            wall=time.perf_counter()-tick
            (root/f'{name}.process.json').write_text(json.dumps(dict(returncode=code,wall_seconds=wall,
                gpu_before=before,gpu_after=gpu()),indent=2))
            if code:raise RuntimeError(f'{name} failed; inspect its log')
            result[variant]=json.loads(path.read_text())['result']
            result[variant]['process_wall_seconds']=wall
            print('END',name,round(wall,2),flush=True)
        native,flash=result['native'],result['flashoptim']
        if [r['step'] for r in native['validation']] != [r['step'] for r in flash['validation']]:
            raise AssertionError('Validation coverage differs')
        difference=max(abs(x['loss']-y['loss']) for x,y in zip(native['validation'],flash['validation']))
        row=dict(round=index,native_step_ms=native['median_step_ms'],flash_step_ms=flash['median_step_ms'],
            step_speedup=native['median_step_ms']/flash['median_step_ms'],
            process_wall_speedup=native['process_wall_seconds']/flash['process_wall_seconds'],
            native_peak_bytes=native['peak_allocated_bytes'],flash_peak_bytes=flash['peak_allocated_bytes'],
            native_state_bytes=native['optimizer_state_bytes'],flash_state_bytes=flash['optimizer_state_bytes'],
            max_validation_loss_difference=difference,passed=difference<=.01)
        rounds.append(row)
        report=dict(plan=plan,rounds=rounds,complete=len(rounds)==3,
            all_loss_gates_passed=all(r['passed'] for r in rounds),
            median_step_speedup=statistics.median(r['step_speedup'] for r in rounds),
            median_process_wall_speedup=statistics.median(r['process_wall_speedup'] for r in rounds))
        (root/'summary.json').write_text(json.dumps(report,indent=2))
    print('CONFIRMATION_FINISHED',str(root),flush=True)


if __name__=='__main__':main()
