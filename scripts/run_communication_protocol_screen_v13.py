"""Isolated NCCL transport/protocol diagnostic; no global environment changes.

No candidate is adopted from this microbenchmark. All protocols must pass
full-coordinate FP64-reference AVG/SUM checks; full training gates follow.
"""
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
    out.mkdir();(out/'attempts').mkdir()
    roles=[dict(name='auto_first',env={}),dict(name='simple',env={'NCCL_PROTO':'Simple'}),
        dict(name='sys',env={'NCCL_P2P_LEVEL':'SYS'}),
        dict(name='sys_simple',env={'NCCL_P2P_LEVEL':'SYS','NCCL_PROTO':'Simple'}),dict(name='auto_last',env={})]
    files=[Path(__file__),root/'scripts/benchmark_fsdp_communication_v13.py',root/'scripts/research_python.sh',root/'scripts/run_vocab_resource_queue_v12.py',root/'training/fsdp/communication_v13.py']
    sources={str(f):sha(f) for f in files};core=source_snapshot(root)
    dump(out/'manifest.json',dict(sources=sources,core_snapshot=core,roles=roles,full_training_qualified=False,
        policy='Fixed order; first clean attempt only. Scoped child environment. Module data never establishes training speedup.'))
    history=[];dump(out/'attempts.json',history)
    for role in roles:
        while True:
            dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease',role=role['name']))
            with stable_lease([0,1,2,3]):
                for path,digest in sources.items():assert sha(path)==digest,path
                assert source_snapshot(root)==core and not apps()
                folder=out/'attempts'/f'{len(history)+1:03d}_{role["name"]}';folder.mkdir()
                env=os.environ.copy();env.pop('NCCL_PROTO',None);env.pop('NCCL_P2P_LEVEL',None)
                env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',
                    NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,ENV,TUNING',
                    NCCL_DEBUG_FILE=str(folder/'nccl.%h.%p.log'),**role['env'])
                env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                command=['bash',str(root/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(root/'scripts/benchmark_fsdp_communication_v13.py'),'--native-only','--output',str(folder/'run')]
                dump(out/'state.json',dict(pid=os.getpid(),stage='running',role=role['name']))
                monitor=run_attempt(command,env,folder,core)
            dirty=bool(monitor['foreign_process_seen'] or monitor['observation_errors'])
            history.append(dict(role=role['name'],folder=str(folder.relative_to(out)),resource_contaminated=dirty,
                returncode=monitor['returncode'],monitor_sha256=sha(folder/'monitor.json'),log_sha256=sha(folder/'worker.log')))
            dump(out/'attempts.json',history)
            if dirty:continue
            assert monitor['returncode']==0,folder
            result=json.loads((folder/'run/result.json').read_text());assert result['complete']
            (out/role['name']).symlink_to(folder.relative_to(out),target_is_directory=True)
            break
    for path,digest in sources.items():assert sha(path)==digest,path
    assert source_snapshot(root)==core
    rows={role['name']:json.loads((out/role['name']/'run/result.json').read_text()) for role in roles}
    dump(out/'result.json',dict(complete=True,full_training_qualified=False,
        rows={name:dict(environment=r['environment'],rows=r['ranks'][0]['rows'],
            all_rank_numerics_passed=all(e['passed'] for rank in r['ranks'] for row in rank['rows'] for e in row['errors'].values()),
            result_sha256=sha(out/name/'run/result.json')) for name,r in rows.items()},
        scope='Microbenchmark of AVG/SUM FP32 RS. No AG/optimizer/model timing or adoption claim.'))
    dump(out/'state.json',dict(pid=os.getpid(),stage='complete_module_only'))


if __name__=='__main__':main()
