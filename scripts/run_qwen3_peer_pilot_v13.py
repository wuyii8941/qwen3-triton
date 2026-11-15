"""Full8B feasibility pilot, including original399 first-gradient references.

Nine updates are not full qualification: no full validation yet. The original
all-coordinate gradient gates are unchanged. The recent paired endpoint short
controls are used only to decide whether to invest in fresh formal controls.
"""
import argparse,json,os,statistics
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from run_vocab_resource_queue_v12 import run_attempt,dump
from run_vocab_screen_v12 import command
from training.provenance import source_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--micro',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();micro=a.micro.resolve();control=a.control.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    m=json.loads((micro/'accepted/run/result.json').read_text());assert m['complete']
    monitor=json.loads((micro/'accepted/monitor.json').read_text())
    assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
    for rank in m['ranks']:
        assert rank['lossless_negative_check_passed'] and rank['repeated_changed_data_check_passed']
        assert all(c['lossless'] for row in rank['rows'] for c in row.get('communication_reports',[]))
    at=lambda mode,order:next(r['median'] for r in m['ranks'][0]['rows'] if r['input_numel']==192946432 and r['mode']==mode and r['order']==order)
    assert all(1-at('peer_bf16',o)/at('native',o)>=.03 for o in [1,2])
    old=json.loads((control/'short_screen_review.json').read_text());assert old['complete'] and old['audit_passed']
    reference=Path('/data1/tzh/proj/qwen_operator_lab/artifacts/triton_qwen3_v1/official_8b_paired_quality_ablation_control/hf_reference_r1')
    archive=micro.parent/'symmetric_rs_micro_002_sources/receipt.json'
    archived=json.loads(archive.read_text());assert archived['complete']
    for row in archived['sources']:assert sha(row['archived'])==row['sha256']
    files=[Path(__file__),root/'scripts/train_qwen3_peer_rs_v13.py',root/'scripts/train_qwen3_vocab_v12.py',root/'scripts/fsdp_peer_rs_v13.py',
        root/'scripts/symmetric_rs_v13.py',root/'scripts/vocab_model_v12.py',root/'scripts/research_python.sh',root/'scripts/run_vocab_resource_queue_v12.py',
        root/'scripts/run_vocab_screen_v12.py',root/'configs/fsdp_8b.json',root/'results/qwen3_triton_v10_gemm_plan_001.json',reference/'result.json',
        micro/'accepted/run/result.json',micro/'accepted/monitor.json',archive,control/'short_screen_review.json',
        *sorted((root/'qwen3_triton').rglob('*.py'))]
    sources={str(p):sha(p) for p in files};core=source_snapshot(root)
    out.mkdir();dump(out/'manifest.json',dict(sources=sources,core_snapshot=core,full_training_qualified=False,
        external_memory_reserve_bytes=512*2**20,effective_memory_cap_GiB=41.875,
        gradient_transport='Bit-checked lossless BF16 staging, FP32 accumulation and output, native FP32 for other sizes.',
        quality=3,warmup=2,measured=4,updates=9,original_reference_dir=str(reference),
        minimum_reduction=.03,policy='Feasibility only. Fresh complete two-order controls/full validation required for adoption. First clean attempt; clean technical/numerical failure is preserved, not timing-selected.'))
    history=[];dump(out/'attempts.json',history)
    def verify():
        for path,digest in sources.items():assert sha(path)==digest,path
        assert source_snapshot(root)==core
    while True:
        dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease'))
        with stable_lease([0,1,2,3]):
            verify();folder=out/f'attempt_{len(history)+1:03d}';folder.mkdir()
            cmd=command(dict(variant='triton_optimized',vocab_ce=True,fused_dw=True),folder/'run')
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
            dump(out/'state.json',dict(pid=os.getpid(),stage='clean_technical_failure',attempt=len(history)))
            raise RuntimeError('Clean pilot failure; inspect preserved log')
        result=json.loads((folder/'run/result.json').read_text());assert result['complete'] and result['screen_only']
        samples=[r['wall_seconds'] for r in result['ranks'][0]['rows'] if r['measured']];assert len(samples)==4
        gradients=[];memory=[];transport=[]
        for row in result['ranks']:
            assert row['updates']==9 and row['parameter_count']==row['optimizer_state_count']==399 and row['all_master_grad_moment_finite_fp32']
            assert len(row['gradient_comparisons'])==1 and len(row['gradient_comparisons'][0]['rows'])==399
            gradients.append(row['gradient_comparisons'][0]['numerical_gate_passed'])
            memory.append(max(row['effective_training_peak'].values())<=41.875)
            transport.append(row['communication']['lossless'] and row['communication']['lossless_violation_flag']==0 and row['communication']['guard_positive_negative_selftest_passed'])
        losses=[r['global_loss'] for r in result['ranks'][0]['rows']]
        ref_losses=json.loads((control/'control_first/run/result.json').read_text())['ranks'][0]['rows']
        drift=[abs(x-y['global_loss']) for x,y in zip(losses,ref_losses)]
        gains=[1-statistics.median(samples)/old['rows'][c]['median'] for c in ['control_first','control_last']]
        report=dict(complete=True,full_training_qualified=False,median=statistics.median(samples),samples=samples,
            all399_reference_gradient_gates=gradients,effective_memory_gates=memory,lossless_transport_gates=transport,
            loss_drift=dict(mean=statistics.mean(drift),maximum=max(drift)),reductions_vs_recent_short_controls=gains,
            eligible_for_formal=all(gradients+memory+transport) and statistics.mean(drift)<.002 and max(drift)<.01 and min(gains)>=.03,
            result_sha256=sha(folder/'run/result.json'),scope='Pilot only; trace retained but not yet audited here. No full validation or adoption claim.')
        (out/'accepted').symlink_to(folder.name,target_is_directory=True);dump(out/'result.json',report)
        dump(out/'state.json',dict(pid=os.getpid(),stage='complete_pilot_only',eligible_for_formal=report['eligible_for_formal']))
        print(json.dumps(report,indent=2),flush=True);break


if __name__=='__main__':main()
