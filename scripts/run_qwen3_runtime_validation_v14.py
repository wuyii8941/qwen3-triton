"""Full512-block validation feasibility, not16-update formal qualification."""
import argparse,json,os,shutil
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from training.provenance import source_snapshot
from run_vocab_resource_queue_v12 import run_attempt,dump
from qwen3_runtime_v14_review import ARMS,run_review,trace_review


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();inputs=Path('/data1/tzh/proj/qwen_operator_lab')
    pilot=a.pilot.resolve();prior=json.loads((pilot/'result.json').read_text())
    assert prior['complete'] and prior['eligible_for_formal'] and json.loads((pilot/'state.json').read_text())['stage']=='complete_pilot_only'
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    reference=inputs/'artifacts/triton_qwen3_v1/official_8b_paired_quality_ablation_control/hf_reference_r1'
    files=[Path(__file__),root/'scripts/train_qwen3_runtime_v14.py',root/'scripts/train_qwen3_vocab_v12.py',root/'scripts/vocab_model_v12.py',
        root/'scripts/qwen3_runtime_v14_review.py',root/'scripts/runtime_trace_v14.py',root/'scripts/qwen3_runtime_v14_contract.py',root/'scripts/qwen3_peer_v13_contract.py',
        root/'scripts/qwen3_vocab_v12_contract.py',root/'scripts/qwen3_training_v11_contract.py',root/'scripts/run_vocab_resource_queue_v12.py',
        root/'scripts/research_python.sh',root/'configs/fsdp_8b.json',root/'results/qwen3_triton_v10_gemm_plan_001.json',
        inputs/'artifacts/official_qwen3_inputs_v1.json',reference/'result.json',pilot/'result.json',pilot/'manifest.json',pilot/'accepted/monitor.json',
        *sorted((root/'qwen3_triton').rglob('*.py')),*sorted((root/'training').rglob('*.py'))]
    sources={str(f):sha(f) for f in files};core=source_snapshot(root);out.mkdir(parents=True);(out/'attempts').mkdir()
    dump(out/'manifest.json',dict(source_root=str(root),sources=sources,core_snapshot=core,pilot=str(pilot),
        model='Qwen/Qwen3-8B',quality_steps=1,warmup=1,measured=2,total_optimizer_updates=5,
        validation_blocks=512,validation_points=[0,1],validation_batch_per_rank=4,full_model_qualified=False,
        purpose='Full-size validation including optimizer states and complete validation GPU trace; numerical full-cohort review remains required.'))
    history=[];plan=json.loads((root/'results/qwen3_triton_v10_gemm_plan_001.json').read_text())
    def verify():
        assert source_snapshot(root)==core
        for path,digest in sources.items():assert sha(path)==digest,path
    while True:
        dump(out/'state.json',dict(pid=os.getpid(),stage='waiting_stable_four_GPU_lease'))
        with stable_lease([0,1,2,3]):
            verify()
            if shutil.disk_usage(out).free<700*2**20:
                dump(out/'state.json',dict(pid=os.getpid(),stage='disk_preflight_failure_no_GPU_run'))
                raise RuntimeError('Insufficient free disk for full raw validation trace; no GPU run started')
            folder=out/'attempts'/f'{len(history)+1:03d}';folder.mkdir()
            cmd=['bash',str(root/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                str(root/'scripts/train_qwen3_runtime_v14.py'),'--runtime','full','--output',str(folder/'run'),
                '--input-root',str(inputs),'--protocol',str(root/'configs/fsdp_8b.json'),
                '--input-manifest',str(inputs/'artifacts/official_qwen3_inputs_v1.json'),'--reference-raw-dir',str(reference),
                '--quality-steps','1','--warmup-steps','1','--measure-steps','2','--full-validation','--trace']
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',GLOO_SOCKET_IFNAME='lo',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
            for key in ['PYTORCH_CUDA_ALLOC_CONF','NCCL_PROTO','NCCL_P2P_LEVEL']:env.pop(key,None)
            dump(out/'state.json',dict(pid=os.getpid(),stage='running',attempt=len(history)+1))
            monitor=run_attempt(cmd,env,folder,core);verify()
        dirty=bool(monitor['foreign_process_seen'] or monitor['observation_errors'])
        row=dict(number=len(history)+1,folder=str(folder.relative_to(out)),resource_contaminated=dirty,returncode=monitor['returncode'],
            accepted=False,monitor_sha256=sha(folder/'monitor.json'),log_sha256=sha(folder/'worker.log'))
        history.append(row);dump(out/'attempts.json',history)
        if dirty:continue
        try:
            assert monitor['returncode']==0,'Clean technical GPU failure'
            d=json.loads((folder/'run/result.json').read_text())
            checks=run_review(d,dict(arm='full_runtime'),formal=False,reference_sha=sha(reference/'result.json'))
            traces=trace_review(folder/'run','full',plan,validation=True)
            assert all(checks['gates'].values()) and all(r['passed'] for r in traces.values()),dict(checks=checks,traces=traces)
        except Exception:
            dump(out/'state.json',dict(pid=os.getpid(),stage='clean_technical_failure'));raise
        row.update(accepted=True,result_sha256=sha(folder/'run/result.json'));dump(out/'attempts.json',history)
        (out/'accepted').symlink_to(folder.relative_to(out),target_is_directory=True)
        dump(out/'result.json',dict(complete=True,eligible_for_formal=True,full_model_qualified=False,checks=checks,traces=traces,
            scope='Five real8B updates and full512 validation at0/1; formal16quality and paired ablations remain pending.'))
        dump(out/'state.json',dict(pid=os.getpid(),stage='complete_validation_feasibility_only'));break


if __name__=='__main__':main()
