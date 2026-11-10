"""Reverse-order confirmation of the remaining two exploratory shapes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('artifacts/optimization_matrix_v1')
    progress=json.loads((root/'progress.json').read_text())
    if not progress.get('complete'):raise RuntimeError('Initial matrix must finish first')
    progress['initial_end_time']=progress['end_time']
    progress['complete']=False
    progress['extension_started']=time.time()
    progress['extension_source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,
        ['training/tied_embedding.py','training/multi_tensor_flashoptim.py','benchmarks/benchmark_tied_embedding_step.py'])}
    for batch,seq in ((1,128),(2,512)):
        shape=f'b{batch}_s{seq}'
        gate=Path('artifacts')/f'optimization_matrix_gate_{shape}_v1'/'result.json'
        for variant in ('both','multi','candidate','baseline'):
            name=f'r1_{shape}_{variant}';output=root/f'{name}.json'
            if output.exists():raise FileExistsError(output)
            progress['active']=name
            (root/'progress.json').write_text(json.dumps(progress,indent=2))
            start=time.time()
            with (root/f'{name}.log').open('w') as stream:
                result=subprocess.run([sys.executable,'-m','benchmarks.benchmark_tied_embedding_step',
                    '--single-stage',variant,'--seq',str(seq),'--batch',str(batch),'--gate',str(gate),
                    '--output',str(output)],stdout=stream,stderr=subprocess.STDOUT)
            progress['jobs'].append(dict(name=name,status='passed' if result.returncode==0 else 'failed',
                seconds=time.time()-start,output=str(output),returncode=result.returncode))
            (root/'progress.json').write_text(json.dumps(progress,indent=2))
            print('CONFIRM_SHAPE',name,result.returncode,flush=True)
            if result.returncode:raise RuntimeError('Stop on failed confirmation')
    progress.pop('active',None);progress.update(complete=True,end_time=time.time())
    (root/'progress.json').write_text(json.dumps(progress,indent=2))


if __name__=='__main__':main()
