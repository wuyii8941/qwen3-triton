"""Run one source-frozen four-GPU runtime check with existing resource lease."""
import argparse,json,os
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from training.provenance import source_snapshot
from run_vocab_resource_queue_v12 import run_attempt,dump


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['adamw','collectives','fsdp','statistics'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    script=root/f'scripts/check_triton_{a.kind}_v14.py';assert script.exists()
    dependencies={'adamw':['adamw.py','memory.py'],'collectives':['collectives.py','peer_memory.py','memory.py'],'fsdp':['collectives.py','peer_memory.py','memory.py','ops.py','dispatch.py','fsdp.py','adamw.py'],
        'statistics':['metrics.py','dispatch.py','memory.py','ops.py']}
    files=[Path(__file__),script,root/'scripts/research_python.sh',root/'scripts/run_vocab_resource_queue_v12.py',root/'qwen3_triton/runtime/__init__.py',
        *(root/'qwen3_triton/runtime'/name for name in dependencies[a.kind])]
    if a.kind=='fsdp':files.extend(sorted((root/'qwen3_triton/kernels').glob('*.py')));files.append(root/'qwen3_triton/overwrite_allocation.py')
    sources={str(f):sha(f) for f in files};out.mkdir(parents=True);(out/'attempts').mkdir()
    dump(out/'manifest.json',dict(kind=a.kind,sources=sources,source_root=str(root),
        numerical_contract={'adamw':'Per-coordinate p/m/v tolerance2e-6 relative plus1e-8/1e-12/1e-14 absolute; independent CPU oracle; restore/skip/multiple updates.',
            'collectives':'All-coordinate CPU oracles for gathers/exchange/SUM/AVG/MAX/MIN; exact BF16-input actual192946432-element RS; system barrier/streams/negative guards.',
            'fsdp':'Real FSDP forward/backward/update and all GPU data operations; native reference and strict complete-step kernel audit.',
            'statistics':'32batch weighted FP64 NLL/count accumulation and final division bitwise vs CPU; FP64 fills, zero-token NaN, initial factories and pure Triton GPU trace.'}[a.kind],
        resource_policy='First clean attempt only; retain contamination, retry after120s idle; clean technical failure stops. No external process signalled.',full_model_qualified=False))
    history=[]
    def verify():
        for name,digest in sources.items():assert sha(name)==digest,name
    while True:
        dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease',kind=a.kind))
        with stable_lease([0,1,2,3]):
            verify();folder=out/'attempts'/f'{len(history)+1:03d}';folder.mkdir();run=folder/'run';run.mkdir()
            cmd=['bash',str(root/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(script),'--output',str(run)]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',GLOO_SOCKET_IFNAME='lo')
            dump(out/'state.json',dict(pid=os.getpid(),stage='running',kind=a.kind,attempt=len(history)+1))
            monitor=run_attempt(cmd,env,folder,source_snapshot(root));verify()
        dirty=bool(monitor['foreign_process_seen'] or monitor['observation_errors'])
        row=dict(number=len(history)+1,resource_contaminated=dirty,returncode=monitor['returncode'],accepted=False,monitor_sha256=sha(folder/'monitor.json'),log_sha256=sha(folder/'worker.log'))
        history.append(row)
        if dirty:dump(out/'attempts.json',history);continue
        if monitor['returncode']:
            dump(out/'attempts.json',history);dump(out/'state.json',dict(pid=os.getpid(),stage='clean_technical_failure',kind=a.kind));raise RuntimeError('Inspect retained real-GPU failure before changing code')
        result=json.loads((run/'check.json').read_text());assert result['complete'] and result['passed'] and result['kind']==a.kind
        row.update(accepted=True,result_sha256=sha(run/'check.json'));dump(out/'attempts.json',history)
        (out/'accepted').symlink_to(folder.relative_to(out),target_is_directory=True)
        dump(out/'state.json',dict(pid=os.getpid(),stage='complete_module_only',kind=a.kind,full_model_qualified=False));print(json.dumps(dict(complete=True,kind=a.kind,full_model_qualified=False)),flush=True);break


if __name__=='__main__':main()
