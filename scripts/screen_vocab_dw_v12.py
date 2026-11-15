"""Pin and monitor the standalone weight-gradient fusion screen."""
from pathlib import Path
import argparse,json,os
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.monitor import run

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();assert out.is_relative_to(Path('/data1/tzh')) and not out.exists();out.mkdir()
    files=[Path(__file__),ROOT/'scripts/benchmark_vocab_dw_v12.py',ROOT/'scripts/research_python.sh',
           ROOT/'qwen3_triton/kernels/vocab_dw.py',ROOT/'qwen3_triton/kernels/linear.py',
           ROOT/'qwen3_triton/kernels/vocab_ce.py',ROOT/'qwen3_triton/kernels/gemm_tuning.py']
    sources={str(p):sha(p) for p in files}
    (out/'manifest.json').write_text(json.dumps(dict(sources=sources,full_model_qualified=False),indent=2)+'\n')
    def state(stage):
        a=dict(pid=os.getpid(),stage=stage);(out/'state.json').write_text(json.dumps(a)+'\n');print(json.dumps(a),flush=True)
    state('waiting_stable_four_GPU_lease')
    with stable_lease([0,1,2,3]):
        for path,digest in sources.items():assert sha(path)==digest,path
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        state('running_weight_gradient_fusion_screen')
        with (out/'worker.log').open('w') as log:run(['bash',str(ROOT/'scripts/research_python.sh'),str(ROOT/'scripts/benchmark_vocab_dw_v12.py'),
            '--output',str(out/'result.json')],cwd=ROOT,env=env,stdout=log,monitor=out/'gpu.json')
        m=json.loads((out/'gpu.json').read_text());assert not m['foreign_process_seen'] and not m['observation_errors']
        for path,digest in sources.items():assert sha(path)==digest,path
    state('complete_module_screen_only')


if __name__=='__main__':main()
