"""Frozen two-order actual36layer8B runtime/optimizer ablation, first clean only."""
import argparse,json,os,subprocess
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from training.provenance import source_snapshot
from run_vocab_resource_queue_v12 import run_attempt,dump
from qwen3_runtime_v14_review import ARMS,run_review,trace_review


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-screen',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();inputs=Path('/data1/tzh/proj/qwen_operator_lab')
    screen=a.validation_screen.resolve();sr=json.loads((screen/'result.json').read_text())
    assert sr['complete'] and sr['eligible_for_formal'] and not sr['full_model_qualified']
    assert json.loads((screen/'state.json').read_text())['stage']=='complete_validation_feasibility_only'
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    reference=inputs/'artifacts/triton_qwen3_v1/official_8b_paired_quality_ablation_control/hf_reference_r1'
    policy=out.parent/'full_runtime_policy_001.json';pd=json.loads(policy.read_text())
    assert pd['total_optimizer_updates']==28 and pd['validation_points']==[0,16]
    assert pd['effective_cap_GiB']==41.875 and pd['performance_adoption_minimum_each_order']==.03
    assert pd['first_order']==list(ARMS) and pd['second_order']==list(reversed(ARMS))
    roles=[dict(arm=arm,order=order,name=f'{arm}_r{order}') for order,key in [(1,'first_order'),(2,'second_order')] for arm in pd[key]]
    # A successful screen from an earlier implementation is not sufficient.
    # Recheck execution bytes plus raw GPU traces and all recorded original
    # error arithmetic with the current CPU reviewer before reserving GPUs.
    sm=json.loads((screen/'manifest.json').read_text())
    for path,digest in sm['sources'].items():
        source=Path(path)
        execution=(source.is_relative_to(root/'qwen3_triton') or source.is_relative_to(root/'training') or source in [
            root/'scripts/train_qwen3_runtime_v14.py',root/'scripts/train_qwen3_vocab_v12.py',root/'scripts/vocab_model_v12.py',
            root/'scripts/research_python.sh',root/'configs/fsdp_8b.json',root/'results/qwen3_triton_v10_gemm_plan_001.json',
            inputs/'artifacts/official_qwen3_inputs_v1.json'])
        if execution:assert sha(source)==digest,source
    monitor=json.loads((screen/'accepted/monitor.json').read_text())
    assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
    plan=json.loads((root/'results/qwen3_triton_v10_gemm_plan_001.json').read_text())
    screened=json.loads((screen/'accepted/run/result.json').read_text())
    screen_checks=run_review(screened,dict(arm='full_runtime'),formal=False,reference_sha=sha(reference/'result.json'))
    screen_traces=trace_review(screen/'accepted/run','full',plan,validation=True)
    assert all(screen_checks['gates'].values()) and all(t['passed'] for t in screen_traces.values())
    for phase,t in screen_traces.items():assert t['trace_sha256']==sr['traces'][phase]['trace_sha256']
    files=[Path(__file__),root/'scripts/train_qwen3_runtime_v14.py',root/'scripts/train_qwen3_vocab_v12.py',root/'scripts/vocab_model_v12.py',
        root/'scripts/review_qwen3_runtime_full_v14.py',root/'scripts/runtime_trace_v14.py',root/'scripts/qwen3_runtime_v14_review.py',root/'scripts/qwen3_runtime_v14_contract.py',
        root/'scripts/qwen3_peer_v13_contract.py',root/'scripts/qwen3_vocab_v12_contract.py',root/'scripts/qwen3_training_v11_contract.py',
        root/'scripts/fsdp_peer_rs_v13.py',root/'scripts/symmetric_rs_v13.py',root/'scripts/research_python.sh',
        root/'scripts/run_vocab_resource_queue_v12.py',root/'configs/fsdp_8b.json',root/'results/qwen3_triton_v10_gemm_plan_001.json',
        policy,inputs/'artifacts/official_qwen3_inputs_v1.json',reference/'result.json',screen/'result.json',screen/'manifest.json',
        screen/'accepted/monitor.json',screen/'accepted/run/result.json',screen/'attempts.json',*sorted((root/'qwen3_triton').rglob('*.py')),*sorted((root/'training').rglob('*.py'))]
    sources={str(f):sha(f) for f in files};core=source_snapshot(root);out.mkdir(parents=True);(out/'attempts').mkdir()
    dump(out/'manifest.json',dict(source_root=str(root),input_root=str(inputs),sources=sources,core_snapshot=core,roles=roles,
        reference_dir=str(reference),reference_result_sha256=sha(reference/'result.json'),policy=str(policy),policy_sha256=sha(policy),
        input_identity=json.loads((reference/'result.json').read_text())['input_manifest_sha256'],
        quality_steps=16,warmup=3,measured=8,real_profile_updates=1,total_updates=28,validation_blocks=512,validation_points=[0,16],
        effective_cap_GiB=41.875,minimum_reduction_each_order=.03,full_model_qualified=False,
        validation_screen=str(screen),prerequisite_reaudit=dict(execution_sources_current=True,checks=screen_checks,traces=screen_traces),
        resource_policy='Fixed eight roles in forward/reverse order. First clean attempt only, retain all contamination and retry after120s idle. Clean technical/gate failure stops. No timing retries.',
        qualification='Original399gradients/forward,16quality,full512validation,memory,complete phase GPU purity and separate performance rule. No default/main branch change.'))
    history=[];dump(out/'attempts.json',history);plan=json.loads((root/'results/qwen3_triton_v10_gemm_plan_001.json').read_text())
    def verify():
        assert source_snapshot(root)==core
        for path,digest in sources.items():assert sha(path)==digest,path
    for role in roles:
        while True:
            dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease',role=role['name']))
            with stable_lease([0,1,2,3]):
                verify();folder=out/'attempts'/f'{len(history)+1:03d}_{role["name"]}';folder.mkdir()
                runtime,variant=ARMS[role['arm']]
                cmd=['bash',str(root/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                    str(root/'scripts/train_qwen3_runtime_v14.py'),'--runtime',runtime,'--variant',variant,'--output',str(folder/'run'),
                    '--input-root',str(inputs),'--protocol',str(root/'configs/fsdp_8b.json'),
                    '--input-manifest',str(inputs/'artifacts/official_qwen3_inputs_v1.json'),'--reference-raw-dir',str(reference),
                    '--quality-steps','16','--warmup-steps','3','--measure-steps','8','--full-validation','--trace']
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',GLOO_SOCKET_IFNAME='lo',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
                for key in ['PYTORCH_CUDA_ALLOC_CONF','NCCL_PROTO','NCCL_P2P_LEVEL']:env.pop(key,None)
                dump(out/'state.json',dict(pid=os.getpid(),stage='running',role=role['name'],attempt=len(history)+1))
                monitor=run_attempt(cmd,env,folder,core);verify()
            dirty=bool(monitor['foreign_process_seen'] or monitor['observation_errors'])
            row=dict(number=len(history)+1,role=role['name'],folder=str(folder.relative_to(out)),resource_contaminated=dirty,
                returncode=monitor['returncode'],accepted=False,monitor_sha256=sha(folder/'monitor.json'),log_sha256=sha(folder/'worker.log'))
            history.append(row);dump(out/'attempts.json',history)
            if dirty:continue
            try:
                assert monitor['returncode']==0,'Clean technical GPU failure'
                d=json.loads((folder/'run/result.json').read_text());assert d['input_manifest_sha256']==json.loads((reference/'result.json').read_text())['input_manifest_sha256']
                checks=run_review(d,role,formal=True,reference_sha=sha(reference/'result.json'))
                traces=trace_review(folder/'run',runtime,plan,validation=True)
                assert all(checks['gates'].values()) and all(r['passed'] for r in traces.values()),dict(checks=checks,traces=traces)
            except Exception:
                dump(out/'state.json',dict(pid=os.getpid(),stage='clean_technical_failure',role=role['name']));raise
            row.update(accepted=True,result_sha256=sha(folder/'run/result.json'),checks=checks,traces=traces);dump(out/'attempts.json',history)
            (out/role['name']).symlink_to(folder.relative_to(out),target_is_directory=True)
            dump(out/'state.json',dict(pid=os.getpid(),stage='accepted_clean_role',role=role['name'],accepted=sum(r['accepted'] for r in history)));break
    verify();dump(out/'state.json',dict(pid=os.getpid(),stage='GPU_complete_independent_CPU_review_pending',accepted=8,full_model_qualified=False))
    cpu=os.environ.copy();cpu['CUDA_VISIBLE_DEVICES']=''
    subprocess.run(['bash',str(root/'scripts/research_python.sh'),str(root/'scripts/review_qwen3_runtime_full_v14.py'),'--output',str(out)],cwd=root,env=cpu,check=True)
    report=json.loads((out/'independent_review.json').read_text());assert report['complete'] and report['audit_passed']
    verify();dump(out/'state.json',dict(pid=os.getpid(),stage='complete_independently_reviewed',full_model_qualified=report['full_model_qualified'],performance_adopted=report['performance_adopted']))


if __name__=='__main__':main()
