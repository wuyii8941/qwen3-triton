"""Screen communication and proven overwrite-only buffers on full Qwen3-8B.

Uses fresh optimizer/model processes, fixed order, and first clean attempts.
Short screens never qualify numerical equivalence or final performance.
"""
import argparse,json,os,statistics,time
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.gpu_lease import apps
from run_vocab_resource_queue_v12 import run_attempt,dump
from run_vocab_screen_v12 import command
from training.provenance import source_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol-screen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.output.resolve();parent=args.protocol_screen.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    # A specific, currently live controller is awaited; absence before result
    # is terminal and never causes an automatic restart.
    pending=json.loads((parent/'state.json').read_text());pid=pending['pid']
    print(json.dumps(dict(pid=os.getpid(),stage='waiting_for_protocol_controller',protocol_pid=pid)),flush=True)
    process_stat=Path(f'/proc/{pid}/stat')
    initial=process_stat.read_text().rsplit(') ',1)[1].split() if process_stat.exists() else None
    while not (parent/'result.json').exists():
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError('Protocol controller ended without a result')
        fields=process_stat.read_text().rsplit(') ',1)[1].split()
        if fields[0] in ('Z','X') or initial is None or fields[19]!=initial[19]:
            raise RuntimeError('Protocol controller is terminal or PID was reused')
        time.sleep(5)
    protocol=json.loads((parent/'result.json').read_text());assert protocol['complete']
    pm=json.loads((parent/'manifest.json').read_text())
    for path,digest in pm['sources'].items():assert sha(path)==digest,path
    eligible=[]
    full_bytes=192946432*4
    at=lambda name,order:next(r['median_seconds'] for r in protocol['rows'][name]['rows'] if r['input_bytes']==full_bytes and r['order']==order)
    for name in ['simple','sys','sys_simple']:
        assert protocol['rows'][name]['all_rank_numerics_passed']
        gains=[1-at(name,o)/at(control,o) for o in [1,2] for control in ['auto_first','auto_last']]
        if min(gains)>=.03:eligible.append((statistics.mean([at(name,1),at(name,2)]),name))
    selected=min(eligible)[1] if eligible else None
    env_choice=next(r['env'] for r in pm['roles'] if r['name']==selected) if selected else {}
    roles=[dict(name='control_first',communication='native',compute=False,buffers=False,env={}),
        dict(name='separate',communication='separate',compute=False,buffers=False,env={}),
        dict(name='compute',communication='native',compute=True,buffers=False,env={}),
        dict(name='buffers',communication='native',compute=False,buffers=True,env={}),
        dict(name='compute_buffers',communication='native',compute=True,buffers=True,env={})]
    if selected:
        roles.extend([dict(name='protocol',communication='native',compute=False,buffers=False,env=env_choice),
            dict(name='protocol_compute_buffers',communication='native',compute=True,buffers=True,env=env_choice)])
    roles.append(dict(name='control_last',communication='native',compute=False,buffers=False,env={}))
    files=[Path(__file__),root/'scripts/train_qwen3_communication_v13.py',root/'scripts/train_qwen3_vocab_v12.py',root/'scripts/vocab_model_v12.py',
        root/'scripts/overwrite_fsdp_v13.py',root/'scripts/check_overwrite_outputs_v13.py',root/'scripts/research_python.sh',root/'scripts/run_vocab_resource_queue_v12.py',
        root/'scripts/run_vocab_screen_v12.py',root/'configs/fsdp_8b.json',root/'results/qwen3_triton_v10_gemm_plan_001.json',
        parent/'result.json',parent/'manifest.json',*sorted((root/'qwen3_triton').rglob('*.py'))]
    sources={str(f):sha(f) for f in files};core=source_snapshot(root)
    out.mkdir();(out/'attempts').mkdir()
    manifest=dict(sources=sources,core_snapshot=core,roles=roles,selected_protocol=selected,
        protocol_selection='>=3% faster than BOTH auto endpoint controls in BOTH within-process orders at full decoder size; FP64 reference passes. Then full-model screen.',
        full_training_qualified=False,memory_cap_GiB=41.875,updates=9,quality=3,warmup=2,measured=4,
        adoption='Require original full399 gradients, full validation, two orders and >=3% complete-step reduction; this screen cannot adopt.')
    dump(out/'manifest.json',manifest);history=[];dump(out/'attempts.json',history)
    def verify():
        for path,digest in sources.items():assert sha(path)==digest,path
        assert source_snapshot(root)==core
    def environment(extra):
        env=os.environ.copy();env.pop('NCCL_PROTO',None);env.pop('NCCL_P2P_LEVEL',None)
        env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',**extra)
        env.pop('PYTORCH_CUDA_ALLOC_CONF',None);return env
    def execute(name,cmd,extra):
        while True:
            dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease',role=name))
            with stable_lease([0,1,2,3]):
                verify();assert not apps();folder=out/'attempts'/f'{len(history)+1:03d}_{name}';folder.mkdir()
                actual=[str(folder/'run') if s=='OUTPUT' else s for s in cmd]
                if name=='overwrite_check':actual[-1]=str(folder/'check.json')
                dump(out/'state.json',dict(pid=os.getpid(),stage='running',role=name))
                monitor=run_attempt(actual,environment(extra),folder,core);verify()
            dirty=bool(monitor['foreign_process_seen'] or monitor['observation_errors'])
            history.append(dict(role=name,folder=str(folder.relative_to(out)),resource_contaminated=dirty,returncode=monitor['returncode'],
                monitor_sha256=sha(folder/'monitor.json'),log_sha256=sha(folder/'worker.log')))
            dump(out/'attempts.json',history)
            if dirty:continue
            assert monitor['returncode']==0,folder
            (out/name).symlink_to(folder.relative_to(out),target_is_directory=True)
            return folder
    execute('overwrite_check',['bash',str(root/'scripts/research_python.sh'),str(root/'scripts/check_overwrite_outputs_v13.py'),'--output','OUTPUT'],{})
    rows={}
    for role in roles:
        cmd=command(dict(variant='triton_optimized',vocab_ce=True,fused_dw=True),'OUTPUT')
        cmd[cmd.index(str(root/'scripts/train_qwen3_vocab_v12.py'))]=str(root/'scripts/train_qwen3_communication_v13.py')
        cmd.extend(['--communication',role['communication'],'--trace'])
        if role['compute']:cmd.append('--overwrite-triton-outputs')
        if role['buffers']:cmd.append('--overwrite-communication-buffers')
        folder=execute(role['name'],cmd,role['env'])
        a=json.loads((folder/'run/result.json').read_text());assert a['complete'] and a['screen_only']
        assert all(r['updates']==9 and r['all_master_grad_moment_finite_fp32'] and r['parameter_count']==r['optimizer_state_count']==399 for r in a['ranks'])
        times=[r['wall_seconds'] for r in a['ranks'][0]['rows'] if r['measured']];assert len(times)==4
        rows[role['name']]=dict(role=role,median=statistics.median(times),samples=times,
            losses=[r['global_loss'] for r in a['ranks'][0]['rows']],
            allocated_GiB=max(r['measured_training_peak']['allocated_gib'] for r in a['ranks']),
            reserved_GiB=max(r['measured_training_peak']['reserved_gib'] for r in a['ranks']),
            result_sha256=sha(folder/'run/result.json'))
        dump(out/'partial_results.json',rows)
    candidates=[]
    for name,row in rows.items():
        gains=[1-row['median']/rows[c]['median'] for c in ['control_first','control_last']]
        drift=[abs(x-y) for x,y in zip(row['losses'],rows['control_first']['losses'])]
        row['quality']=dict(mean=statistics.mean(drift),maximum=max(drift))
        row['reductions_vs_endpoint_controls']=gains
        row['eligible_for_formal']=name not in ['control_first','control_last'] and min(gains)>=.03 and statistics.mean(drift)<.002 and max(drift)<.01 and max(row['allocated_GiB'],row['reserved_GiB'])<=41.875
        if row['eligible_for_formal']:candidates.append(row)
    selected_model=min(candidates,key=lambda r:r['median'])['role']['name'] if candidates else None
    verify();dump(out/'result.json',dict(complete=True,full_training_qualified=False,selected=selected_model,rows=rows,
        scope='Nine updates, no original-reference gradient or full-validation check. Actual traces retained; no final speedup/adoption claim.'))
    dump(out/'state.json',dict(pid=os.getpid(),stage='complete_screen_only',selected=selected_model))


if __name__=='__main__':main()
