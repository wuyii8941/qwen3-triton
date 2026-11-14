"""Compare HF+Flash, accepted V9 and explicit GEMM plans in both orders.

Six runs (or eight with a second plan) preserve V9's 27-update/full-validation protocol.
Reuse the original lossless native first-gradient archive and require a strict
native control in each new HF arm. No new dense archive is generated.
"""
from pathlib import Path
import argparse,json,os,shutil,subprocess,time
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.gpu_lease import apps
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.monitor import run
from training.provenance import source_snapshot
from training.input_manifest import validate

ROOT=Path(__file__).resolve().parents[1]


def roles(compare=False):
    forward=('hf','v9','v10_all','v10') if compare else ('hf','v9','v10')
    return [dict(name=f'{arm}_r{order}',arm=arm,order=order,
                 variant='hf' if arm=='hf' else 'triton_optimized',tuned=arm.startswith('v10'))
            for order,arms in ((1,forward),(2,tuple(reversed(forward)))) for arm in arms]


def command(args,role):
    result=['bash',str(ROOT/'scripts/research_python.sh'),'-m','torch.distributed.run',
        '--standalone','--nproc_per_node=4',str(ROOT/'scripts/train_qwen3_triton_8b.py'),
        '--variant',role['variant'],'--output',str(args.output/role['name']),
        '--protocol',str(args.protocol),'--input-manifest',str(args.input_manifest),
        '--input-root',str(args.input_root),
        '--quality-steps','16','--warmup-steps','3','--measure-steps','8',
        '--full-validation','--trace','--reference-raw-dir',str(args.reference)]
    if role['tuned']:
        plan=args.compare_gemm_plan if role['arm']=='v10_all' else args.gemm_plan
        result.extend(['--gemm-plan',str(plan)])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--input-manifest',type=Path,required=True)
    parser.add_argument('--input-root',type=Path,default=ROOT)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--gemm-plan',type=Path,required=True)
    parser.add_argument('--module-check',type=Path,required=True)
    parser.add_argument('--compare-gemm-plan',type=Path)
    parser.add_argument('--compare-module-check',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if bool(args.compare_gemm_plan)!=bool(args.compare_module_check):parser.error('Both comparison arguments are required together')
    for key in ('protocol','input_manifest','input_root','reference','gemm_plan','module_check','output'):
        setattr(args,key,getattr(args,key).resolve())
    if args.compare_gemm_plan:
        args.compare_gemm_plan=args.compare_gemm_plan.resolve()
        args.compare_module_check=args.compare_module_check.resolve()
    queue=roles(bool(args.compare_gemm_plan))
    assert not args.output.is_relative_to(Path('/home/tzh'))
    if args.dry_run:
        print(json.dumps(dict(GPU_started=False,roles=[dict(**role,command=command(args,role)) for role in queue]),indent=2));return
    # Check relative path binding and every frozen model/token byte on CPU,
    # before admission or allocating CUDA contexts on the four workers.
    previous_cwd=Path.cwd()
    try:
        os.chdir(args.input_root)
        input_identity=validate(args.input_manifest,args.protocol,json.loads(args.protocol.read_text()))
    finally:os.chdir(previous_cwd)
    modules=[(args.gemm_plan,args.module_check)]
    if args.compare_gemm_plan:modules.append((args.compare_gemm_plan,args.compare_module_check))
    for plan_path,module_path in modules:
        module=json.loads((module_path/'measurements/result.json').read_text())
        monitor=json.loads((module_path/'gpu.json').read_text())
        assert module['complete'] and len(module['rows'])==49
        assert sum(r.get('local_VJP_passed',False) for r in module['rows'])==7
        assert monitor['selected_device_clean'] and monitor['returncode']==0
        assert module['plan']['sha256']==sha(plan_path)
        for path,h in module['sources'].items():assert sha(ROOT/path)==h
    reference=json.loads((args.reference/'result.json').read_text())
    assert input_identity==reference['input_manifest_sha256']
    assert reference['variant']=='hf' and reference['world_size']==4 and reference['layers']==36
    assert reference['math_policy']['allow_bf16_reduced_precision_reduction'] is False
    for rank in range(4):assert sha(args.reference/f'raw_rank{rank}.pt.gz')==reference['ranks'][rank]['reference_archive']['sha256']
    args.output.mkdir(parents=True,exist_ok=False)
    assert shutil.disk_usage(args.output).free>3*2**30,'Need space for traces; original archives are reused'
    files=[Path(__file__),ROOT/'scripts/train_qwen3_triton_8b.py',
        ROOT/'scripts/review_qwen3_tuned_suite.py',args.protocol,args.input_manifest,args.gemm_plan,
        args.reference/'result.json',args.module_check/'measurements/result.json',
        args.module_check/'gpu.json',*sorted((ROOT/'qwen3_triton').rglob('*.py'))]
    if args.compare_gemm_plan:
        files.extend([args.compare_gemm_plan,args.compare_module_check/'measurements/result.json',args.compare_module_check/'gpu.json'])
    plan_map={'v10':dict(path=str(args.gemm_plan),sha256=sha(args.gemm_plan),selection='MLP and LM head')}
    if args.compare_gemm_plan:plan_map['v10_all']=dict(path=str(args.compare_gemm_plan),sha256=sha(args.compare_gemm_plan),selection='All selected operator shapes')
    manifest=dict(sources={str(p):sha(p) for p in files},core_snapshot=source_snapshot(ROOT),
        roles=queue,reference_dir=str(args.reference),reference_result_sha256=sha(args.reference/'result.json'),
        plans=plan_map,input_root=str(args.input_root),input_manifest_sha256=input_identity,
        gemm_plan=str(args.gemm_plan),gemm_plan_sha256=sha(args.gemm_plan),
        quality_steps=16,warmup_steps=3,measure_steps=8,original_V9_commit='13dbdf984201a70ca4bba5fb38c72606cd33fbeb',
        performance_adopted=False,scope=__doc__)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def state(stage):
        (args.output/'queue_state.json').write_text(json.dumps(dict(pid=os.getpid(),stage=stage))+'\n')
        print(stage,flush=True)
    state('waiting_stable_four_GPU_lease')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',
        PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    with stable_lease([0,1,2,3]):
        for role in queue:
            assert not apps()
            for path,h in manifest['sources'].items():assert sha(path)==h
            assert source_snapshot(ROOT)==manifest['core_snapshot']
            state('running_'+role['name'])
            with (args.output/(role['name']+'.log')).open('x') as log:
                run(command(args,role),cwd=ROOT,env=env,stdout=log,monitor=args.output/(role['name']+'.gpu.json'))
            monitor=json.loads((args.output/(role['name']+'.gpu.json')).read_text())
            assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
            result=json.loads((args.output/role['name']/'result.json').read_text())
            assert result['complete'] and result['variant']==role['variant']
            assert all(r['updates']==27 and r['parameter_count']==399 for r in result['ranks'])
            expected=plan_map[role['arm']]['sha256'] if role['tuned'] else None
            actual=result['gemm_plan']['sha256'] if result['gemm_plan'] else None
            assert actual==expected
            owned=set(monitor['known_owned_pids'])
            for _ in range(30):
                live=apps()
                if not live:break
                assert all(pid in owned for gpu,pid in live);time.sleep(1)
            assert not apps()
        for path,h in manifest['sources'].items():assert sha(path)==h
    state('GPU_complete_independent_CPU_review')
    subprocess.run(['bash',str(ROOT/'scripts/research_python.sh'),str(ROOT/'scripts/review_qwen3_tuned_suite.py'),
                    '--output',str(args.output)],cwd=ROOT,check=True)
    state('complete_reviewed_no_automatic_adoption')


if __name__=='__main__':main()
