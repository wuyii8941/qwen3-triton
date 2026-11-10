"""Independent, fixed-budget strong-baseline screen; not convergence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('artifacts/flashoptim_strong_baselines_v1');root.mkdir(exist_ok=False)
    variants=['native','triton','compile_native','flashoptim','compile_triton','compile_flashoptim']
    plan=dict(variants=variants,steps=256,seq=512,lr=1e-5,rounds=2,
              scope='Independent 256-step screen, not convergence. Compilation included; shared GPU.',
              gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    (root/'plan.json').write_text(json.dumps(plan,indent=2));results=[]
    for round_id in range(2):
        for variant in (variants if round_id==0 else list(reversed(variants))):
            name=f'r{round_id}_{variant}';print('START',name,flush=True)
            start=time.perf_counter()
            with (root/f'{name}.log').open('x') as log:
                code=subprocess.run([sys.executable,'-m','benchmarks.screen_research_candidates',
                    '--mode','train','--variant',variant,'--steps','256','--output',str(root/f'{name}.json')],
                    stdout=log,stderr=subprocess.STDOUT).returncode
            results.append(dict(name=name,returncode=code,wall_seconds=time.perf_counter()-start))
            (root/'status.json').write_text(json.dumps(dict(results=results,complete=False),indent=2))
            print('END',name,code,flush=True)
    (root/'status.json').write_text(json.dumps(dict(results=results,complete=True),indent=2))


if __name__=='__main__':main()
