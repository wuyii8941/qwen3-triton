"""Short full8B feasibility/quality/timing screen of vocabulary-local CE."""
from pathlib import Path
import argparse,json,os,statistics
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.gpu_lease import apps
from qwen3_triton.benchmark_utils.monitor import run
from training.provenance import source_snapshot

ROOT=Path(__file__).resolve().parents[1]
INPUT=Path('/data1/tzh/proj/qwen_operator_lab')


def command(role,output,full=False):
    cmd=['bash',str(ROOT/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
         str(ROOT/'scripts/train_qwen3_vocab_v12.py'),'--variant',role['variant'],'--output',str(output),
         '--protocol',str(ROOT/'configs/fsdp_8b.json'),'--input-manifest',str(INPUT/'artifacts/official_qwen3_inputs_v1.json'),
         '--input-root',str(INPUT),'--keep-full-layers','36','--checkpoint-every','1','--skip-embedding-backward-gather',
         '--quality-steps','16' if full else '3','--warmup-steps','3' if full else '2','--measure-steps','8' if full else '4']
    if role['vocab_ce']:cmd.append('--vocab-ce')
    else:cmd.append('--retain-lm-head')
    if role.get('fused_dw'):cmd.append('--fused-vocab-dw')
    if role['variant']=='triton_optimized':cmd.extend(['--gemm-plan',str(ROOT/'results/qwen3_triton_v10_gemm_plan_001.json')])
    if full:cmd.extend(['--full-validation','--trace','--reference-raw-dir',str(INPUT/'artifacts/triton_qwen3_v1/official_8b_paired_quality_ablation_control/hf_reference_r1')])
    return cmd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--module-parent',type=Path,default=INPUT/'artifacts/triton_qwen3_v12/module_001')
    p.add_argument('--fused-vocab-dw',action='store_true');p.add_argument('--trace',action='store_true')
    args=p.parse_args();out=args.output.resolve();assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    parent=args.module_parent.resolve()
    module=json.loads((parent/'module/result.json').read_text());monitor=json.loads((parent/'gpu.json').read_text())
    assert module['complete'] and min(module['reductions'])>.03
    assert module.get('fused_dw',False)==args.fused_vocab_dw
    assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
    for path,digest in json.loads((parent/'manifest.json').read_text())['sources'].items():assert sha(path)==digest,path
    roles=[dict(name='candidate',variant='triton_optimized',vocab_ce=True,fused_dw=args.fused_vocab_dw)]
    if args.fused_vocab_dw:roles.append(dict(name='head_unfused',variant='triton_optimized',vocab_ce=True,fused_dw=False))
    roles.extend([dict(name='triton_control',variant='triton_optimized',vocab_ce=False,fused_dw=False),
                  dict(name='HF_custom_head',variant='hf',vocab_ce=True,fused_dw=args.fused_vocab_dw)])
    files=[Path(__file__),ROOT/'scripts/train_qwen3_vocab_v12.py',ROOT/'scripts/vocab_model_v12.py',
           ROOT/'scripts/research_python.sh',ROOT/'configs/fsdp_8b.json',
           ROOT/'results/qwen3_triton_v10_gemm_plan_001.json',
           INPUT/'artifacts/official_qwen3_inputs_v1.json',parent/'module/result.json',parent/'gpu.json',
           *sorted((ROOT/'qwen3_triton').rglob('*.py'))]
    if args.trace:files.extend([ROOT/'scripts/qwen3_vocab_v12_contract.py',ROOT/'scripts/qwen3_training_v11_contract.py'])
    out.mkdir();manifest=dict(sources={str(p):sha(p) for p in files},roles=roles,core_snapshot=source_snapshot(ROOT),
        minimum_reduction=.03,memory_cap_GiB=41.875,mean_loss_drift_limit=.002,max_loss_drift_limit=.01,
        updates=9,quality_updates=3,warmup=2,measured=4,full_model_qualified=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def state(stage):
        a=dict(pid=os.getpid(),stage=stage);(out/'state.json').write_text(json.dumps(a)+'\n');print(json.dumps(a),flush=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    results={};state('waiting_stable_four_GPU_lease')
    with stable_lease([0,1,2,3]):
        for role in roles:
            assert not apps()
            for path,digest in manifest['sources'].items():assert sha(path)==digest,path
            assert source_snapshot(ROOT)==manifest['core_snapshot']
            name=role['name'];state('running_'+name)
            cmd=command(role,out/name)
            if args.trace:cmd.append('--trace')
            with (out/(name+'.log')).open('w') as log:
                run(cmd,cwd=ROOT,env=env,stdout=log,monitor=out/(name+'.gpu.json'))
            m=json.loads((out/(name+'.gpu.json')).read_text())
            assert not m['foreign_process_seen'] and not m['observation_errors']
            a=json.loads((out/name/'result.json').read_text())
            assert a['complete'] and a['screen_only'] and len(a['ranks'])==4
            samples=[r['wall_seconds'] for r in a['ranks'][0]['rows'] if r['measured']]
            assert len(samples)==4
            assert all(r['updates']==9 and r['all_master_grad_moment_finite_fp32'] and
                       r['parameter_count']==r['optimizer_state_count']==399 and max(r['measured_training_peak'].values())<=41.875 for r in a['ranks'])
            trace=None
            if args.trace:
                from qwen3_vocab_v12_contract import audit_trace
                plan=json.loads((ROOT/'results/qwen3_triton_v10_gemm_plan_001.json').read_text()) if role['variant']=='triton_optimized' else None
                trace=audit_trace(out/name/'training_trace.json',role['variant'],role['vocab_ce'],plan,
                                  fused_dw=role['fused_dw'],dw_config=(64,128,32,8,8,3) if role['fused_dw'] else None)
                assert trace['passed'],trace
            results[name]=dict(median=statistics.median(samples),samples=samples,trace=trace,
                losses=[r['global_loss'] for r in a['ranks'][0]['rows']],
                allocated_GiB=max(r['measured_training_peak']['allocated_gib'] for r in a['ranks']),
                reserved_GiB=max(r['measured_training_peak']['reserved_gib'] for r in a['ranks']),
                result_sha256=sha(out/name/'result.json'),monitor_sha256=sha(out/(name+'.gpu.json')))
    for path,digest in manifest['sources'].items():assert sha(path)==digest,path
    ref=results['triton_control'];quality={}
    for name,a in results.items():
        drift=[abs(x-y) for x,y in zip(a['losses'],ref['losses'])]
        quality[name]=dict(mean=statistics.mean(drift),maximum=max(drift),passed=statistics.mean(drift)<.002 and max(drift)<.01)
    reduction=1-results['candidate']['median']/ref['median']
    if args.fused_vocab_dw:assert results['candidate']['losses']==results['head_unfused']['losses']
    result=dict(complete=True,full_model_qualified=False,results=results,quality=quality,reduction_vs_triton_control=reduction,
        eligible_for_formal=all(q['passed'] for q in quality.values()) and reduction>=.03,
        limitations='Nine-update feasibility screen, no archived-reference gradient comparison or full validation yet. Not an adoption or HF speedup claim.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');state('complete_screen_only')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
