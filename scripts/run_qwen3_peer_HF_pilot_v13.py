"""Test the same peer-RS infrastructure on the native HF/Flash model.

Original tight native-control first-gradient gates and bitwise forward remain
mandatory. This is a nine-update feasibility check, not final qualification.
No timings from failed/over-budget profiles support performance claims.
"""
import argparse,json,os,statistics
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from run_vocab_resource_queue_v12 import run_attempt,dump
from run_vocab_screen_v12 import command
from training.provenance import source_snapshot
from qwen3_peer_v13_contract import audit_trace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--triton-pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();parent=a.triton_pilot.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    prior=json.loads((parent/'result.json').read_text());assert prior['complete'] and prior['eligible_for_formal']
    trace=audit_trace(parent/'accepted/run/training_trace.json','triton_optimized',True,json.loads((root/'results/qwen3_triton_v10_gemm_plan_001.json').read_text()),peer=True)
    assert trace['passed']
    reference=Path('/data1/tzh/proj/qwen_operator_lab/artifacts/triton_qwen3_v1/official_8b_paired_quality_ablation_control/hf_reference_r1')
    old=Path('/data1/tzh/proj/qwen_operator_lab/artifacts/triton_qwen3_v12/formal_003/native_r1/result.json')
    files=[Path(__file__),root/'scripts/train_qwen3_peer_rs_v13.py',root/'scripts/train_qwen3_vocab_v12.py',root/'scripts/fsdp_peer_rs_v13.py',
        root/'scripts/symmetric_rs_v13.py',root/'scripts/vocab_model_v12.py',root/'scripts/research_python.sh',root/'scripts/run_vocab_resource_queue_v12.py',
        root/'scripts/run_vocab_screen_v12.py',root/'scripts/qwen3_peer_v13_contract.py',root/'scripts/qwen3_vocab_v12_contract.py',root/'scripts/qwen3_training_v11_contract.py',
        root/'configs/fsdp_8b.json',reference/'result.json',parent/'result.json',old,*sorted((root/'qwen3_triton').rglob('*.py'))]
    sources={str(p):sha(p) for p in files};core=source_snapshot(root)
    out.mkdir();dump(out/'manifest.json',dict(sources=sources,core_snapshot=core,full_training_qualified=False,
        external_memory_reserve_bytes=512*2**20,effective_memory_cap_GiB=41.875,updates=9,
        reference_dir=str(reference),native_control_rule='Original .0001*refnorm+1e-7 per-parameter gradient gate plus bitwise forward; unchanged.',
        policy='First clean attempt. Clean technical failure stops, preserved as failed feasibility. No timing based retries.'))
    history=[];dump(out/'attempts.json',history)
    def verify():
        for path,digest in sources.items():assert sha(path)==digest,path
        assert source_snapshot(root)==core
    while True:
        dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease'))
        with stable_lease([0,1,2,3]):
            verify();folder=out/f'attempt_{len(history)+1:03d}';folder.mkdir()
            cmd=command(dict(variant='hf',vocab_ce=False,fused_dw=False),folder/'run')
            cmd[cmd.index(str(root/'scripts/train_qwen3_vocab_v12.py'))]=str(root/'scripts/train_qwen3_peer_rs_v13.py')
            cmd.extend(['--trace','--reference-raw-dir',str(reference)])
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
            env.pop('NCCL_PROTO',None);env.pop('NCCL_P2P_LEVEL',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
            dump(out/'state.json',dict(pid=os.getpid(),stage='running',attempt=len(history)+1))
            monitor=run_attempt(cmd,env,folder,core);verify()
        dirty=bool(monitor['foreign_process_seen'] or monitor['observation_errors'])
        history.append(dict(folder=folder.name,resource_contaminated=dirty,returncode=monitor['returncode'],
            monitor_sha256=sha(folder/'monitor.json'),log_sha256=sha(folder/'worker.log')));dump(out/'attempts.json',history)
        if dirty:continue
        if monitor['returncode']:
            dump(out/'state.json',dict(pid=os.getpid(),stage='clean_technical_failure'))
            raise RuntimeError('Clean HF feasibility failure; inspect preserved log')
        result=json.loads((folder/'run/result.json').read_text());assert result['complete'] and result['screen_only']
        reference_result=json.loads(old.read_text());gradient=[];memory=[];lossless=[]
        for row in result['ranks']:
            assert row['updates']==9 and row['parameter_count']==row['optimizer_state_count']==399 and row['all_master_grad_moment_finite_fp32']
            g=row['gradient_comparisons'][0];assert g['strict_native_control'] and len(g['rows'])==399
            gradient.append(g['numerical_gate_passed']);memory.append(max(row['effective_training_peak'].values())<=41.875)
            lossless.append(row['communication']['lossless'] and row['communication']['lossless_violation_flag']==0)
        times=[r['wall_seconds'] for r in result['ranks'][0]['rows'] if r['measured']];assert len(times)==4
        drift=[abs(x['global_loss']-y['global_loss']) for x,y in zip(result['ranks'][0]['rows'],reference_result['ranks'][0]['rows'])]
        cuda=audit_trace(folder/'run/training_trace.json','hf',False,None,peer=True)
        report=dict(complete=True,full_training_qualified=False,median=statistics.median(times),samples=times,
            original_tight_gradient_gates=gradient,effective_memory_gates=memory,lossless_transport_gates=lossless,
            loss_drift=dict(mean=statistics.mean(drift),maximum=max(drift)),actual_CUDA=cuda,
            eligible_for_formal=all(gradient+memory+lossless) and statistics.mean(drift)<.002 and max(drift)<.01 and cuda['passed'],
            result_sha256=sha(folder/'run/result.json'),scope='Nine-update feasibility only, no full-validation or adoption claim.')
        (out/'accepted').symlink_to(folder.name,target_is_directory=True);dump(out/'result.json',report)
        dump(out/'state.json',dict(pid=os.getpid(),stage='complete_pilot_only',eligible_for_formal=report['eligible_for_formal']))
        print(json.dumps(report,indent=2),flush=True);break


if __name__=='__main__':main()
