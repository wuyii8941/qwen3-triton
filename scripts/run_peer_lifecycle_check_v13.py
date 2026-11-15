"""Resource-monitored lifecycle check; clean technical failures are retained."""
import argparse,json,os
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from run_vocab_resource_queue_v12 import run_attempt,dump
from training.provenance import source_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    out.mkdir();files=[Path(__file__),root/'scripts/check_peer_lifecycle_v13.py',root/'scripts/fsdp_peer_rs_v13.py',root/'scripts/symmetric_rs_v13.py',
        root/'scripts/research_python.sh',root/'scripts/run_vocab_resource_queue_v12.py']
    sources={str(p):sha(p) for p in files};core=source_snapshot(root)
    dump(out/'manifest.json',dict(sources=sources,core_snapshot=core,full_training_qualified=False))
    attempt=0
    while True:
        dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease'))
        with stable_lease([0,1,2,3]):
            for path,digest in sources.items():assert sha(path)==digest,path
            assert source_snapshot(root)==core
            attempt+=1;folder=out/f'attempt_{attempt:03d}';folder.mkdir()
            command=['bash',str(root/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(root/'scripts/check_peer_lifecycle_v13.py'),'--output',str(folder/'check.json')]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            env.pop('NCCL_PROTO',None);env.pop('NCCL_P2P_LEVEL',None)
            dump(out/'state.json',dict(pid=os.getpid(),stage='running',attempt=attempt))
            monitor=run_attempt(command,env,folder,core)
        if monitor['foreign_process_seen'] or monitor['observation_errors']:continue
        assert monitor['returncode']==0,folder
        report=json.loads((folder/'check.json').read_text());assert report['complete']
        for path,digest in sources.items():assert sha(path)==digest,path
        (out/'accepted').symlink_to(folder.name,target_is_directory=True)
        dump(out/'state.json',dict(pid=os.getpid(),stage='complete_lifecycle_check',full_training_qualified=False))
        break


if __name__=='__main__':main()
