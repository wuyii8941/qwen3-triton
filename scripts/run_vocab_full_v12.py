"""Freeze and run paired complete8B controls and vocabulary CE candidates."""
from pathlib import Path
import argparse,json,os,subprocess
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.gpu_lease import apps
from qwen3_triton.benchmark_utils.monitor import run
from training.provenance import source_snapshot
from run_vocab_screen_v12 import ROOT,INPUT,command


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--checkpoint-screen',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();screen=args.screen.resolve();cp_screen=args.checkpoint_screen.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    a=json.loads((screen/'result.json').read_text());cp=json.loads((cp_screen/'result.json').read_text())
    assert a['complete'] and a['eligible_for_formal'] and cp['complete']
    for folder in [screen,cp_screen]:
        for path,digest in json.loads((folder/'manifest.json').read_text())['sources'].items():assert sha(path)==digest,path
    every=cp['selected_checkpoint_every'];assert every in (0,1,2)
    arms=[dict(arm='native',variant='hf',vocab_ce=False,checkpoint_every=1),
          dict(arm='triton_control',variant='triton_optimized',vocab_ce=False,checkpoint_every=1)]
    if every!=1:arms.append(dict(arm='head_cp1',variant='triton_optimized',vocab_ce=True,checkpoint_every=1))
    arms.extend([dict(arm='HF_custom_head',variant='hf',vocab_ce=True,checkpoint_every=every),
                 dict(arm='candidate',variant='triton_optimized',vocab_ce=True,checkpoint_every=every)])
    roles=[dict(**a,order=order,name=a['arm']+f'_r{order}') for order,rows in [(1,arms),(2,list(reversed(arms)))] for a in rows]
    reference=INPUT/'artifacts/triton_qwen3_v1/official_8b_paired_quality_ablation_control/hf_reference_r1'
    ref=json.loads((reference/'result.json').read_text())
    files=[*sorted((ROOT/'scripts').glob('*v12.py')),ROOT/'scripts/qwen3_training_v11_contract.py',
           ROOT/'scripts/research_python.sh',ROOT/'configs/fsdp_8b.json',
           ROOT/'results/qwen3_triton_v10_gemm_plan_001.json',INPUT/'artifacts/official_qwen3_inputs_v1.json',
           reference/'result.json',screen/'manifest.json',screen/'result.json',cp_screen/'manifest.json',cp_screen/'result.json',
           *sorted((ROOT/'qwen3_triton').rglob('*.py'))]
    out.mkdir();manifest=dict(roles=roles,checkpoint_every=every,sources={str(p):sha(p) for p in files},
        core_snapshot=source_snapshot(ROOT),source_root=str(ROOT),input_root=str(INPUT),input_identity=ref['input_manifest_sha256'],
        reference_dir=str(reference),reference_result_sha256=sha(reference/'result.json'),
        gemm_plan=dict(path=str(ROOT/'results/qwen3_triton_v10_gemm_plan_001.json'),sha256=sha(ROOT/'results/qwen3_triton_v10_gemm_plan_001.json')),
        memory_cap_GiB=41.875,minimum_reduction_each_order=.03,
        quality_steps=16,warmup=3,measured=8,
        policy='Chosen checkpoint policy fixed by prior short screen. Require >=3% lower complete optimizer-step time vs both matched-order native HF and V11 Triton controls; all399 FP32 gradient gates, full validation and actual CUDA/collectives must pass. HF custom head separately shows portability and implementation attribution. No automatic defaults.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def verify():
        for path,digest in manifest['sources'].items():assert sha(path)==digest,path
        assert source_snapshot(ROOT)==manifest['core_snapshot']
    def state(stage):
        a=dict(pid=os.getpid(),stage=stage);(out/'state.json').write_text(json.dumps(a)+'\n');print(json.dumps(a),flush=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    state('waiting_stable_four_GPU_lease')
    with stable_lease([0,1,2,3]):
        for role in roles:
            assert not apps();verify();name=role['name'];state('running_'+name)
            cmd=command(role,out/name,full=True);cmd[cmd.index('--checkpoint-every')+1]=str(role['checkpoint_every'])
            with (out/(name+'.log')).open('w') as log:run(cmd,cwd=ROOT,env=env,stdout=log,monitor=out/(name+'.gpu.json'))
            m=json.loads((out/(name+'.gpu.json')).read_text())
            assert not m['foreign_process_seen'] and not m['observation_errors']
            result=json.loads((out/name/'result.json').read_text())
            assert result['complete'] and not result['screen_only'] and all(r['updates']==27 for r in result['ranks'])
        verify()
    state('GPU_complete_independent_CPU_review')
    cpu=env.copy();cpu['CUDA_VISIBLE_DEVICES']=''
    subprocess.run(['bash',str(ROOT/'scripts/research_python.sh'),str(ROOT/'scripts/review_vocab_full_v12.py'),
                    '--output',str(out)],cwd=ROOT,env=cpu,check=True)
    a=json.loads((out/'independent_review.json').read_text());assert a['complete'] and a['audit_passed']
    verify();state('complete_reviewed_no_default_change')
    print(json.dumps(dict(complete=True,qualified_training_improvement=a['qualified_training_improvement']),indent=2),flush=True)


if __name__=='__main__':main()
