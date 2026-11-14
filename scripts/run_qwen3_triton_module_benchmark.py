"""Capture complete8B reference inputs, time96modulecases and independently review.

Isolated forward/backward timings exclude FSDP/NCCL/optimizer. This controller
uses the same shared GPU admission lock and strict contamination monitor as
the paired training suite. It does not adopt model optimizations.
"""
from pathlib import Path
import argparse,json,os,subprocess
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.gpu_lease import apps
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.monitor import run
from training.provenance import source_snapshot

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/fsdp_8b.json')
    parser.add_argument('--input-manifest',type=Path,default=ROOT/'artifacts/official_qwen3_inputs_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve();args.protocol=args.protocol.resolve();args.input_manifest=args.input_manifest.resolve()
    if args.output.is_relative_to(Path('/home/tzh')):parser.error('Output under /home/tzh is forbidden')
    command=['bash',str(ROOT/'scripts/research_python.sh'),str(ROOT/'scripts/benchmark_qwen3_triton_modules.py'),'--protocol',str(args.protocol),'--input-manifest',str(args.input_manifest),'--output',str(args.output/'measurements')]
    if args.plan:
        print(json.dumps(dict(model='Qwen/Qwen3-8B',command=command,cases=96,GPU_started=False),indent=2));return
    args.output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'scripts/benchmark_qwen3_triton_modules.py',ROOT/'scripts/review_qwen3_triton_modules.py',args.protocol,args.input_manifest,*sorted((ROOT/'qwen3_triton').rglob('*.py'))]
    manifest=dict(sources={str(p):sha(p) for p in files},core_snapshot=source_snapshot(ROOT),scope=__doc__,performance_adopted=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('waiting_stable_exclusive_all_four_GPU_lease',flush=True)
    with stable_lease([0,1,2,3]):
        assert not apps()
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        with (args.output/'worker.log').open('x') as log:run(command,cwd=ROOT,env=env,stdout=log,monitor=args.output/'gpu.json')
        monitor=json.loads((args.output/'gpu.json').read_text());assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and monitor['observation_errors']==0
        for path,h in manifest['sources'].items():assert sha(path)==h
        assert source_snapshot(ROOT)==manifest['core_snapshot']
        result=json.loads((args.output/'measurements/result.json').read_text());assert result['complete'] and len(result['rows'])==96
    subprocess.run(['bash',str(ROOT/'scripts/research_python.sh'),str(ROOT/'scripts/review_qwen3_triton_modules.py'),'--output',str(args.output)],cwd=ROOT,check=True)
    print('complete_public_actual8B_module_timing_and_review_not_adopted',flush=True)

if __name__=='__main__':main()
