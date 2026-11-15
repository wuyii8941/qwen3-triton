"""Source-frozen full36layer8B feasibility run, not formal adoption."""
import argparse,json,os,statistics
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from training.provenance import source_snapshot
from run_vocab_resource_queue_v12 import run_attempt,dump
from qwen3_runtime_v14_contract import audit_trace
from runtime_trace_v14 import locate_trace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();inputs=Path('/data1/tzh/proj/qwen_operator_lab')
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    reference=inputs/'artifacts/triton_qwen3_v1/official_8b_paired_quality_ablation_control/hf_reference_r1'
    policy=inputs/'artifacts/triton_qwen3_runtime_v14/full_runtime_policy_001.json'
    files=[Path(__file__),root/'scripts/train_qwen3_runtime_v14.py',root/'scripts/train_qwen3_vocab_v12.py',root/'scripts/vocab_model_v12.py',
        root/'scripts/qwen3_runtime_v14_contract.py',root/'scripts/runtime_trace_v14.py',root/'scripts/qwen3_peer_v13_contract.py',root/'scripts/qwen3_vocab_v12_contract.py',root/'scripts/qwen3_training_v11_contract.py',
        root/'scripts/research_python.sh',root/'scripts/run_vocab_resource_queue_v12.py',root/'configs/fsdp_8b.json',
        root/'results/qwen3_triton_v10_gemm_plan_001.json',policy,inputs/'artifacts/official_qwen3_inputs_v1.json',reference/'result.json',
        *sorted((root/'qwen3_triton').rglob('*.py')),*sorted((root/'training').rglob('*.py'))]
    sources={str(f):sha(f) for f in files};core=source_snapshot(root);out.mkdir(parents=True);(out/'attempts').mkdir()
    dump(out/'manifest.json',dict(source_root=str(root),sources=sources,core_snapshot=core,model='Qwen/Qwen3-8B',layers=36,world_size=4,
        quality=1,warmup=1,measured=2,real_profile_updates=1,total_updates=5,validation_blocks=0,
        archived399_gradient_check=True,full_model_qualified=False,resource_policy='First clean attempt only; retain and retry contamination after120s idle; clean technical failure stops.',
        acceptance='Complete real8B5update screen, original399gradients, FP32 states, effective memory and exact initial/full-step/setup GPU coverage. No formal16quality/full512validation or speed adoption claim.'))
    history=[]
    def verify():
        for path,digest in sources.items():assert sha(path)==digest,path
        assert source_snapshot(root)==core
    while True:
        dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease'))
        with stable_lease([0,1,2,3]):
            verify();folder=out/'attempts'/f'{len(history)+1:03d}';folder.mkdir()
            cmd=['bash',str(root/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                str(root/'scripts/train_qwen3_runtime_v14.py'),'--runtime','full','--output',str(folder/'run'),
                '--input-root',str(inputs),'--protocol',str(root/'configs/fsdp_8b.json'),
                '--input-manifest',str(inputs/'artifacts/official_qwen3_inputs_v1.json'),'--reference-raw-dir',str(reference),
                '--quality-steps','1','--warmup-steps','1','--measure-steps','2','--trace']
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                GLOO_SOCKET_IFNAME='lo',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
            env.pop('PYTORCH_CUDA_ALLOC_CONF',None);env.pop('NCCL_PROTO',None);env.pop('NCCL_P2P_LEVEL',None)
            dump(out/'state.json',dict(pid=os.getpid(),stage='running',attempt=len(history)+1))
            monitor=run_attempt(cmd,env,folder,core);verify()
        dirty=bool(monitor['foreign_process_seen'] or monitor['observation_errors'])
        row=dict(number=len(history)+1,resource_contaminated=dirty,returncode=monitor['returncode'],accepted=False,monitor_sha256=sha(folder/'monitor.json'),log_sha256=sha(folder/'worker.log'))
        history.append(row)
        if dirty:dump(out/'attempts.json',history);continue
        if monitor['returncode']:
            dump(out/'attempts.json',history);dump(out/'state.json',dict(pid=os.getpid(),stage='clean_technical_failure'));raise RuntimeError('Preserved clean full8B failure; inspect before changing code')
        d=json.loads((folder/'run/result.json').read_text());assert d['complete'] and d['screen_only'] and not d['full_runtime_qualified']
        assert all(r['updates']==5 and r['parameter_count']==r['optimizer_state_count']==399 and r['all_master_grad_moment_finite_fp32'] for r in d['ranks'])
        gates=dict(gradients=all(r['gradient_comparisons'][0]['numerical_gate_passed'] for r in d['ranks']),
            memory=all(max(r['effective_training_peak'].values())<=41.875 and max(r['effective_all_phase_peak'].values())<=41.875 for r in d['ranks']))
        plan=json.loads((root/'results/qwen3_triton_v10_gemm_plan_001.json').read_text())
        traces={phase:audit_trace(locate_trace(folder/'run',name),phase=phase,plan=plan) for phase,name in [
            ('setup','setup_trace.json'),('initial_step','initial_step_trace.json'),('full_step','full_step_trace.json')]}
        gates['all_GPU_data_compute_Triton']=all(t['passed'] for t in traces.values())
        assert all(gates.values()),dict(gates=gates,traces=traces)
        row.update(accepted=True,result_sha256=sha(folder/'run/result.json'));dump(out/'attempts.json',history)
        (out/'accepted').symlink_to(folder.relative_to(out),target_is_directory=True)
        result=dict(complete=True,eligible_for_formal=True,full_model_qualified=False,gates=gates,traces=traces,
            median_seconds=statistics.median(x['wall_seconds'] for x in d['ranks'][0]['rows'] if x['measured']),
            scope='Five real full8B updates and complete phase traces; not16quality/fullvalidation qualification or a performance adoption.')
        dump(out/'result.json',result);dump(out/'state.json',dict(pid=os.getpid(),stage='complete_pilot_only'))
        print(json.dumps(dict(complete=True,gates=gates,full_model_qualified=False)),flush=True);break


if __name__=='__main__':main()
