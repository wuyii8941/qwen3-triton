"""Explicit diagnostic wrapper; does not change the normal timing worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

import torch


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',type=Path,required=True)
    args,_=parser.parse_known_args()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Diagnostic requires explicit deterministic cuBLAS workspace')
    files=[Path(__file__).resolve(),Path(__file__).with_name('compare_fsdp_training.py').resolve()]
    def hashes():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    initial=hashes()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    runpy.run_path(str(files[1]),run_name='__main__')
    if hashes()!=initial or not torch.are_deterministic_algorithms_enabled():
        raise ValueError('Diagnostic source/settings changed')
    if int(os.environ['RANK'])==0:
        path=args.output/'result.json';report=json.loads(path.read_text())
        report['deterministic_diagnostic']=dict(enabled=True,source_hashes=initial,
            cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            scope='Short loss-trajectory diagnostic; not ordinary steady-state performance')
        path.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
