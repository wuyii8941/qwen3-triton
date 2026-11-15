"""Run a bounded symmetric-memory availability probe under the shared GPU lease."""
from pathlib import Path
import argparse,json,os
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from run_vocab_resource_queue_v12 import run_attempt,dump
from qwen3_triton.gradient_archive import sha
from training.provenance import source_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    out.mkdir();sources={str(f):sha(f) for f in [Path(__file__),root/'scripts/probe_symmetric_reduce_v13.py',root/'scripts/run_vocab_resource_queue_v12.py',root/'scripts/research_python.sh']}
    core=source_snapshot(root);dump(out/'manifest.json',dict(sources=sources,core_snapshot=core,full_training_qualified=False))
    dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease'))
    with stable_lease([0,1,2,3]):
        for path,digest in sources.items():assert sha(path)==digest,path
        assert source_snapshot(root)==core
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        env.pop('NCCL_PROTO',None);env.pop('NCCL_P2P_LEVEL',None)
        command=['bash',str(root/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(root/'scripts/probe_symmetric_reduce_v13.py'),'--output',str(out/'probe.json')]
        dump(out/'state.json',dict(pid=os.getpid(),stage='running'))
        monitor=run_attempt(command,env,out,core)
    clean=not monitor['foreign_process_seen'] and not monitor['observation_errors']
    if not clean:
        dump(out/'state.json',dict(pid=os.getpid(),stage='resource_contaminated_no_result_adopted'));return
    assert monitor['returncode']==0,out
    report=json.loads((out/'probe.json').read_text());assert report['complete']
    for path,digest in sources.items():assert sha(path)==digest,path
    dump(out/'state.json',dict(pid=os.getpid(),stage='complete_availability_probe',available=report['available']))


if __name__=='__main__':main()
