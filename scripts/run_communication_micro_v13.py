"""Resource-monitored entry for the topology-aware FP32 RS microbenchmark."""
import argparse,json,os
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.gpu_lease import apps
from run_vocab_resource_queue_v12 import run_attempt,dump
from training.provenance import source_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.output.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    out.mkdir();files=[Path(__file__),root/'scripts/benchmark_fsdp_communication_v13.py',root/'scripts/research_python.sh',root/'scripts/run_vocab_resource_queue_v12.py',root/'training/fsdp/communication_v13.py']
    sources={str(f):sha(f) for f in files};core=source_snapshot(root)
    dump(out/'manifest.json',dict(sources=sources,core_snapshot=core,full_training_qualified=False,
        policy='Keep first clean result, preserve dirty attempts; numerical/technical failure stops. No timing-based retries.'))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',
        NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,ENV,TUNING',NCCL_DEBUG_FILE=str(out/'nccl.%h.%p.log'))
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    attempt=0
    while True:
        dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease'))
        with stable_lease([0,1,2,3]):
            for path,digest in sources.items():assert sha(path)==digest,path
            assert source_snapshot(root)==core and not apps()
            attempt+=1;folder=out/f'attempt_{attempt:03d}';folder.mkdir()
            dump(out/'state.json',dict(pid=os.getpid(),stage='running',attempt=attempt))
            command=['bash',str(root/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(root/'scripts/benchmark_fsdp_communication_v13.py'),'--output',str(folder/'run')]
            monitor=run_attempt(command,env,folder,core)
        if monitor['foreign_process_seen'] or monitor['observation_errors']:continue
        assert monitor['returncode']==0,folder
        result=json.loads((folder/'run/result.json').read_text());assert result['complete']
        for path,digest in sources.items():assert sha(path)==digest,path
        assert source_snapshot(root)==core
        (out/'accepted').symlink_to(folder.name,target_is_directory=True)
        dump(out/'state.json',dict(pid=os.getpid(),stage='complete_module_only',accepted=folder.name))
        print(json.dumps(dict(complete=True,accepted=folder.name,full_training_qualified=False)),flush=True)
        break


if __name__=='__main__':main()
