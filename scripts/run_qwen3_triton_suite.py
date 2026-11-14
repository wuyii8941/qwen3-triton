"""Run/review paired full Qwen3-8B training and all five ablations.

Sixteen fresh four-rank processes use the same weights, inputs and math
policy. The reverse order repeats every arm. No-fallback audits and full
validation are excluded from measured updates. Failed numerical gates are
retained; a completed run never automatically adopts an optimization.
"""
from pathlib import Path
import argparse,json,os,shutil,subprocess,time
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.gpu_lease import apps
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.monitor import run
from training.provenance import source_snapshot

ROOT=Path(__file__).resolve().parents[1]

def roles():
    names=['triton_base','triton_optimized','ablate_attention','ablate_norm_rope','ablate_mlp','ablate_linear','ablate_loss']
    result=[dict(name='hf_reference_r1',variant='hf',order=1)]
    result.extend(dict(name=n+'_r1',variant=n,order=1) for n in names)
    result.extend(dict(name=n+'_r2',variant=n,order=2) for n in reversed(names))
    result.append(dict(name='hf_reference_r2',variant='hf',order=2))
    return result

def command(args,role):
    result=['bash',str(ROOT/'scripts/research_python.sh'),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(ROOT/'scripts/train_qwen3_triton_8b.py'),'--variant',role['variant'],'--output',str(args.output/role['name']),'--protocol',str(args.protocol),'--input-manifest',str(args.input_manifest),'--quality-steps','16','--warmup-steps','3','--measure-steps','8','--full-validation','--trace']
    if role['name']=='hf_reference_r1':result.append('--save-reference')
    else:result.extend(['--reference-raw-dir',str(args.output/'hf_reference_r1')])
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/fsdp_8b.json')
    parser.add_argument('--input-manifest',type=Path,default=ROOT/'artifacts/official_qwen3_inputs_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',type=int,nargs=4,default=[0,1,2,3])
    parser.add_argument('--plan',action='store_true',help='Print all commands without creating files, acquiring GPUs or running training')
    args=parser.parse_args()
    if len(set(args.gpus))!=4 or min(args.gpus)<0:parser.error('Select four distinct nonnegative physical GPU indices')
    args.output=args.output.resolve();args.protocol=args.protocol.resolve();args.input_manifest=args.input_manifest.resolve()
    if args.output.is_relative_to(Path('/home/tzh')):parser.error('Output under /home/tzh is forbidden')
    queue=roles()
    if args.plan:
        print(json.dumps(dict(model='Qwen/Qwen3-8B',roles=[dict(**role,command=command(args,role)) for role in queue],GPU_started=False),indent=2));return
    args.output.mkdir(parents=True,exist_ok=False)
    assert shutil.disk_usage(args.output).free>34*2**30,'Need24GiB capped reference archives plus10GiB reserve'
    files=[Path(__file__),ROOT/'scripts/train_qwen3_triton_8b.py',ROOT/'scripts/review_qwen3_triton_suite.py',args.protocol,args.input_manifest,*sorted((ROOT/'qwen3_triton').rglob('*.py'))]
    manifest=dict(sources={str(p):sha(p) for p in files},core_snapshot=source_snapshot(ROOT),roles=queue,quality_steps=16,warmup_steps=3,measure_steps=8,validation_batch_per_rank=4,validation_scope='Every complete frozen validation block before and after16updates',strict_BF16_pointwise_qualification=False,performance_adopted=False,scope=__doc__)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def state(stage):
        (args.output/'queue_state.json').write_text(json.dumps(dict(pid=os.getpid(),stage=stage))+'\n');print(stage,flush=True)
    state('waiting_stable_four_GPU_lease')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,args.gpus)),PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    with stable_lease(args.gpus):
        reference_sha=None
        for role in queue:
            assert not apps(),'Suite requires no other active compute contexts'
            for path,h in manifest['sources'].items():assert sha(path)==h
            if reference_sha is not None:assert sha(args.output/'hf_reference_r1/result.json')==reference_sha
            state('running_complete8B_'+role['name'])
            with (args.output/(role['name']+'.log')).open('x') as log:
                run(command(args,role),cwd=ROOT,env=env,stdout=log,monitor=args.output/(role['name']+'.gpu.json'))
            monitor=json.loads((args.output/(role['name']+'.gpu.json')).read_text())
            assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and monitor['observation_errors']==0
            result=json.loads((args.output/role['name']/'result.json').read_text())
            assert result['complete'] and result['model']=='Qwen/Qwen3-8B' and result['layers']==36 and result['world_size']==4
            assert all(r['updates']==27 and r['parameter_count']==399 for r in result['ranks'])
            if role['name']=='hf_reference_r1':reference_sha=sha(args.output/'hf_reference_r1/result.json')
            owned=set(monitor['known_owned_pids'])
            for _ in range(30):
                live=apps()
                if not live:break
                assert all(pid in owned for gpu,pid in live);time.sleep(1)
            assert not apps() and shutil.disk_usage(args.output).free>8*2**30
        for path,h in manifest['sources'].items():assert sha(path)==h
        assert source_snapshot(ROOT)==manifest['core_snapshot']
    state('GPU_complete_independent_CPU_review')
    subprocess.run(['bash',str(ROOT/'scripts/research_python.sh'),str(ROOT/'scripts/review_qwen3_triton_suite.py'),'--output',str(args.output)],cwd=ROOT,check=True)
    state('complete_reviewed_adoption_not_automatic')

if __name__=='__main__':main()
