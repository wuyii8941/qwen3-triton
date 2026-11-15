"""Test whether vocabulary-local head memory permits less recomputation."""
from pathlib import Path
import argparse,json,os,statistics,subprocess
from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.gpu_lease import apps
from qwen3_triton.benchmark_utils.monitor import run
from training.provenance import source_snapshot
from run_vocab_screen_v12 import ROOT,command


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();parent=args.parent.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and not out.exists()
    previous=json.loads((parent/'result.json').read_text());pm=json.loads((parent/'manifest.json').read_text())
    assert previous['complete'] and previous['eligible_for_formal']
    for path,digest in pm['sources'].items():assert sha(path)==digest,path
    profiles=[dict(name='cp1_control',checkpoint_every=1),dict(name='cp2',checkpoint_every=2),dict(name='cp0',checkpoint_every=0)]
    sources=dict(pm['sources']);sources[str(Path(__file__).resolve())]=sha(Path(__file__))
    sources[str(parent/'result.json')]=sha(parent/'result.json')
    out.mkdir();manifest=dict(profiles=profiles,sources=sources,core_snapshot=source_snapshot(ROOT),memory_cap_GiB=41.875,
        selection='Choose faster cp0/cp2 only if >=3% faster than fresh cp1 control and mean/max loss drift <.002/.01; otherwise retain cp1.',minimum_reduction=.03)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def state(stage):
        a=dict(pid=os.getpid(),stage=stage);(out/'state.json').write_text(json.dumps(a)+'\n');print(json.dumps(a),flush=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    rows=[];state('waiting_stable_four_GPU_lease')
    with stable_lease([0,1,2,3]):
        for profile in profiles:
            assert not apps()
            for path,digest in sources.items():assert sha(path)==digest,path
            assert source_snapshot(ROOT)==manifest['core_snapshot']
            name=profile['name'];state('running_'+name)
            cmd=command(dict(variant='triton_optimized',vocab_ce=True),out/name)
            cmd[cmd.index('--checkpoint-every')+1]=str(profile['checkpoint_every'])
            failure=None
            try:
                with (out/(name+'.log')).open('w') as log:run(cmd,cwd=ROOT,env=env,stdout=log,monitor=out/(name+'.gpu.json'))
            except subprocess.CalledProcessError as exc:failure=exc.returncode
            monitor=json.loads((out/(name+'.gpu.json')).read_text())
            assert not monitor['foreign_process_seen'] and not monitor['observation_errors']
            if failure:
                assert 'out of memory' in (out/(name+'.log')).read_text().lower()
                rows.append(dict(profile=profile,completed=False,reason='CUDA out of memory',monitor_sha256=sha(out/(name+'.gpu.json')),log_sha256=sha(out/(name+'.log'))))
                continue
            a=json.loads((out/name/'result.json').read_text())
            assert a['complete'] and a['screen_only'] and all(r['updates']==9 and r['all_master_grad_moment_finite_fp32'] and max(r['measured_training_peak'].values())<=41.875 for r in a['ranks'])
            samples=[r['wall_seconds'] for r in a['ranks'][0]['rows'] if r['measured']]
            rows.append(dict(profile=profile,completed=True,median=statistics.median(samples),samples=samples,
                losses=[r['global_loss'] for r in a['ranks'][0]['rows']],
                allocated_GiB=max(r['measured_training_peak']['allocated_gib'] for r in a['ranks']),
                reserved_GiB=max(r['measured_training_peak']['reserved_gib'] for r in a['ranks']),
                result_sha256=sha(out/name/'result.json'),monitor_sha256=sha(out/(name+'.gpu.json'))))
    assert rows[0]['completed'];control=rows[0];eligible=[]
    for row in rows:
        if not row['completed']:continue
        drift=[abs(a-b) for a,b in zip(row['losses'],control['losses'])]
        row['quality']=dict(mean=statistics.mean(drift),maximum=max(drift))
        row['reduction']=1-row['median']/control['median']
        row['eligible']=row['reduction']>=.03 and statistics.mean(drift)<.002 and max(drift)<.01
        if row['eligible']:eligible.append(row)
    selected=min(eligible,key=lambda r:r['median']) if eligible else control
    result=dict(complete=True,rows=rows,selected_checkpoint_every=selected['profile']['checkpoint_every'],
        full_model_qualified=False,scope='Short feasibility screen, no full validation or reference gradient comparison.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');state('complete_screen_only');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
