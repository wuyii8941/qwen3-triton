"""First-clean source-frozen native/Triton reduction-order diagnosis."""
import argparse,json,os
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from training.provenance import source_snapshot
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from run_vocab_resource_queue_v12 import run_attempt,dump


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    files=[Path(__file__),root/'scripts/check_runtime_reduction_order_v14.py',root/'scripts/research_python.sh',root/'scripts/run_vocab_resource_queue_v12.py',
        *sorted((root/'qwen3_triton/runtime').glob('*.py'))]
    sources={str(p):sha(p) for p in files};core=source_snapshot(root);out.mkdir(parents=True);(out/'attempts').mkdir()
    dump(out/'manifest.json',dict(source_root=str(root),sources=sources,core_snapshot=core,full_model_qualified=False,
        purpose='NativeFP32reduction order at actual CE/dX sizes. Diagnostic only, no full8B gate change.'))
    history=[]
    while True:
        dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease'))
        with stable_lease([0,1,2,3]):
            for p,h in sources.items():assert sha(p)==h,p
            folder=out/'attempts'/f'{len(history)+1:03d}';folder.mkdir();(folder/'run').mkdir()
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',GLOO_SOCKET_IFNAME='lo')
            for key in ['NCCL_PROTO','NCCL_P2P_LEVEL']:env.pop(key,None)
            cmd=['bash',str(root/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(root/'scripts/check_runtime_reduction_order_v14.py'),'--output',str(folder/'run')]
            dump(out/'state.json',dict(pid=os.getpid(),stage='running',attempt=len(history)+1));monitor=run_attempt(cmd,env,folder,core)
            for p,h in sources.items():assert sha(p)==h,p
        dirty=bool(monitor['foreign_process_seen'] or monitor['observation_errors'])
        row=dict(number=len(history)+1,resource_contaminated=dirty,returncode=monitor['returncode'],accepted=False,
            monitor_sha256=sha(folder/'monitor.json'),log_sha256=sha(folder/'worker.log'))
        history.append(row);dump(out/'attempts.json',history)
        if dirty:continue
        if monitor['returncode']:
            dump(out/'state.json',dict(pid=os.getpid(),stage='clean_technical_failure'));raise RuntimeError('Preserved clean reduction-order probe failure')
        d=json.loads((folder/'run/check.json').read_text());assert d['complete'] and len(d['ranks'])==4
        row.update(accepted=True,result_sha256=sha(folder/'run/check.json'));dump(out/'attempts.json',history)
        (out/'accepted').symlink_to(folder.relative_to(out),target_is_directory=True)
        dump(out/'state.json',dict(pid=os.getpid(),stage='complete_diagnostic_only'));break


if __name__=='__main__':main()
