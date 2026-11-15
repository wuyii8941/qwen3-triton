"""Freeze paired full8B default, V12, HF-peer and Triton-peer comparisons.

Eight fixed roles, first clean attempt only. GPU completion is not adoption:
independent CPU review must still inspect all original numerical/quality gates,
effective memory, source identity, real CUDA contracts and both timing orders.
"""
import argparse,json,os,time
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from run_vocab_resource_queue_v12 import run_attempt,dump
from run_vocab_screen_v12 import command
from training.provenance import source_snapshot
from qwen3_peer_v13_contract import audit_trace


def completed_pilot(path):
    state=json.loads((path/'state.json').read_text());pid=state['pid'];proc=Path(f'/proc/{pid}/stat')
    initial=proc.read_text().rsplit(') ',1)[1].split()[19] if proc.exists() else None
    while not (path/'result.json').exists():
        fields=proc.read_text().rsplit(') ',1)[1].split() if proc.exists() else None
        if fields is None or fields[0] in ['Z','X'] or fields[19]!=initial:
            raise RuntimeError('Pilot controller ended without a complete result')
        time.sleep(5)
    while json.loads((path/'state.json').read_text())['stage']!='complete_pilot_only':
        fields=proc.read_text().rsplit(') ',1)[1].split() if proc.exists() else None
        if fields is None or fields[0] in ['Z','X']:raise RuntimeError('Pilot ended before reviewed terminal state')
        time.sleep(2)
    result=json.loads((path/'result.json').read_text());assert result['complete'] and result['eligible_for_formal']
    monitor=json.loads((path/'accepted/monitor.json').read_text())
    assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--triton-pilot',type=Path,required=True)
    p.add_argument('--HF-pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();tp=a.triton_pilot.resolve();hp=a.HF_pilot.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    print(json.dumps(dict(pid=os.getpid(),stage='waiting_for_specific_pilot_results')),flush=True)
    triton=completed_pilot(tp);hf=completed_pilot(hp)
    reference=Path('/data1/tzh/proj/qwen_operator_lab/artifacts/triton_qwen3_v1/official_8b_paired_quality_ablation_control/hf_reference_r1')
    policy=out.parent/'adoption_policy_001.json';assert json.loads(policy.read_text())['thresholds_unchanged']
    plan=json.loads((root/'results/qwen3_triton_v10_gemm_plan_001.json').read_text())
    assert audit_trace(tp/'accepted/run/training_trace.json','triton_optimized',True,plan,peer=True)['passed']
    assert audit_trace(hp/'accepted/run/training_trace.json','hf',False,None,peer=True)['passed']
    arms=[dict(arm='native',variant='hf',vocab_ce=False,fused_dw=False,peer=False),
        dict(arm='V12',variant='triton_optimized',vocab_ce=True,fused_dw=True,peer=False),
        dict(arm='HF_peer',variant='hf',vocab_ce=False,fused_dw=False,peer=True),
        dict(arm='candidate',variant='triton_optimized',vocab_ce=True,fused_dw=True,peer=True)]
    roles=[dict(**arm,order=order,name=arm['arm']+f'_r{order}') for order,rows in [(1,arms),(2,list(reversed(arms)))] for arm in rows]
    files=[Path(__file__),root/'scripts/train_qwen3_peer_rs_v13.py',root/'scripts/train_qwen3_vocab_v12.py',root/'scripts/train_qwen3_communication_v13.py',
        root/'scripts/fsdp_peer_rs_v13.py',root/'scripts/symmetric_rs_v13.py',root/'scripts/overwrite_fsdp_v13.py',root/'scripts/vocab_model_v12.py',
        root/'scripts/research_python.sh',root/'scripts/run_vocab_resource_queue_v12.py',root/'scripts/run_vocab_screen_v12.py',
        root/'scripts/qwen3_peer_v13_contract.py',root/'scripts/qwen3_vocab_v12_contract.py',root/'scripts/qwen3_training_v11_contract.py',
        root/'configs/fsdp_8b.json',root/'results/qwen3_triton_v10_gemm_plan_001.json',reference/'result.json',
        Path('/data1/tzh/proj/qwen_operator_lab/artifacts/official_qwen3_inputs_v1.json'),policy,
        tp/'result.json',tp/'manifest.json',tp/'accepted/run/result.json',tp/'accepted/monitor.json',
        hp/'result.json',hp/'manifest.json',hp/'accepted/run/result.json',hp/'accepted/monitor.json',
        *sorted((root/'qwen3_triton').rglob('*.py'))]
    sources={str(f):sha(f) for f in files};core=source_snapshot(root)
    out.mkdir();(out/'attempts').mkdir()
    dump(out/'manifest.json',dict(source_root=str(root),input_root='/data1/tzh/proj/qwen_operator_lab',roles=roles,
        sources=sources,core_snapshot=core,reference_dir=str(reference),reference_result_sha256=sha(reference/'result.json'),
        input_identity=json.loads((reference/'result.json').read_text())['input_manifest_sha256'],
        quality_steps=16,warmup=3,measured=8,updates=27,validation_blocks=512,validation_points=[0,16],
        memory_cap_GiB=41.875,peer_external_reserve_bytes=512*2**20,peer_torch_allocator_cap_GiB=41.375,
        minimum_reduction_each_order=.03,default_changed=False,
        resource_policy='Fixed roles and order; first clean complete attempt wins; contaminated attempts retained and retried after continuous120s idle; clean technical failure stops; no timing-based retries.',
        adoption='All8 roles must pass original399 gradients, forward gates, short quality, full validation, effective memory and CUDA checks. Candidate must reduce whole-step median >=3% vs V12, native HF and HF with the same peer infrastructure in BOTH orders. No defaults switched.'))
    history=[];dump(out/'attempts.json',history)
    def verify():
        for path,digest in sources.items():assert sha(path)==digest,path
        assert source_snapshot(root)==core
    for role in roles:
        while True:
            dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease',role=role['name']))
            with stable_lease([0,1,2,3]):
                verify();folder=out/'attempts'/f'{len(history)+1:03d}_{role["name"]}';folder.mkdir()
                cmd=command(role,folder/'run',full=True)
                entry='train_qwen3_peer_rs_v13.py' if role['peer'] else 'train_qwen3_communication_v13.py'
                cmd[cmd.index(str(root/'scripts/train_qwen3_vocab_v12.py'))]=str(root/'scripts'/entry)
                if not role['peer']:cmd.extend(['--communication','native'])
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
                env.pop('NCCL_PROTO',None);env.pop('NCCL_P2P_LEVEL',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                dump(out/'state.json',dict(pid=os.getpid(),stage='running',role=role['name'],attempt=len(history)+1))
                monitor=run_attempt(cmd,env,folder,core);verify()
            dirty=bool(monitor['foreign_process_seen'] or monitor['observation_errors'])
            row=dict(number=len(history)+1,role=role['name'],folder=str(folder.relative_to(out)),accepted=False,
                resource_contaminated=dirty,returncode=monitor['returncode'],monitor_sha256=sha(folder/'monitor.json'),log_sha256=sha(folder/'worker.log'))
            if dirty:
                history.append(row);dump(out/'attempts.json',history);continue
            if monitor['returncode']:
                history.append(row);dump(out/'attempts.json',history)
                dump(out/'state.json',dict(pid=os.getpid(),stage='clean_technical_failure',role=role['name']))
                raise RuntimeError('Clean full-run failure; inspect retained log, never automatically retry it')
            data=json.loads((folder/'run/result.json').read_text())
            assert data['complete'] and not data['screen_only'] and all(r['updates']==27 for r in data['ranks'])
            cuda=audit_trace(folder/'run/training_trace.json',role['variant'],role['vocab_ce'],plan if role['variant']=='triton_optimized' else None,peer=role['peer'])
            assert cuda['passed'],cuda
            row.update(accepted=True,result_sha256=sha(folder/'run/result.json'),trace_sha256=sha(folder/'run/training_trace.json'),actual_CUDA=cuda)
            history.append(row);dump(out/'attempts.json',history)
            (out/role['name']).symlink_to(folder.relative_to(out),target_is_directory=True)
            dump(out/'state.json',dict(pid=os.getpid(),stage='accepted_clean_role',role=role['name'],accepted=sum(r['accepted'] for r in history)))
            break
    verify();dump(out/'state.json',dict(pid=os.getpid(),stage='GPU_complete_independent_CPU_review_pending',accepted=8,qualified_training_improvement=False))
    print(json.dumps(dict(GPU_complete=True,independent_CPU_review_pending=True,qualified_training_improvement=False)),flush=True)


if __name__=='__main__':main()
