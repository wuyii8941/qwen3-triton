"""Admit, pin and monitor a four-GPU vocabulary CE module screen."""
from pathlib import Path
import argparse,json,os,subprocess
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.monitor import run

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fused-dw',action='store_true')
    args=p.parse_args();output=args.output.resolve();assert not output.exists()
    assert output.is_relative_to(Path('/data1/tzh'));output.mkdir()
    sources={str(p):sha(p) for p in [Path(__file__),ROOT/'scripts/benchmark_vocab_ce_v12.py',
             ROOT/'scripts/research_python.sh',*sorted((ROOT/'qwen3_triton').rglob('*.py'))]}
    (output/'manifest.json').write_text(json.dumps(dict(sources=sources,full_model_qualified=False),indent=2)+'\n')
    def state(stage):
        (output/'state.json').write_text(json.dumps(dict(pid=os.getpid(),stage=stage))+'\n')
        print(stage,flush=True)
    state('waiting_stable_four_GPU_lease')
    with stable_lease([0,1,2,3]):
        for path,digest in sources.items():assert sha(path)==digest,path
        state('running_module_screen')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
        env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        with (output/'worker.log').open('w') as log:
            cmd=['bash',str(ROOT/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                 str(ROOT/'scripts/benchmark_vocab_ce_v12.py'),'--output',str(output/'module')]
            if args.fused_dw:cmd.append('--fused-dw')
            run(cmd,cwd=ROOT,env=env,stdout=log,monitor=output/'gpu.json')
        monitor=json.loads((output/'gpu.json').read_text())
        assert not monitor['foreign_process_seen'] and not monitor['observation_errors']
        for path,digest in sources.items():assert sha(path)==digest,path
    state('complete_module_screen_only')


if __name__=='__main__':main()
