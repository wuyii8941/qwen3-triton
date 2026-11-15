"""Export reviewed full-runtime proof, keeping speed adoption separate.

Only bounded JSON/source text is exported. Weights, tokens, dense gradients,
profiler events, environments and checkpoints remain local. Historical failed
execution sources are recovered by exact SHA from Git or preserved backups.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from training.provenance import source_snapshot


def write_new(path,value):
    payload=(json.dumps(value,indent=2)+'\n').encode()
    if path.exists():
        assert path.read_bytes()==payload,path
        return
    with path.open('xb') as f:f.write(payload)


def historical_sources(root,base):
    cache={};receipts=[];embedded=0
    def recover(path,digest):
        nonlocal embedded
        key=(str(path),digest)
        if key in cache:return cache[key]
        relative=path.relative_to(root).as_posix()
        revisions=subprocess.check_output(['git','log','--format=%H','--',relative],cwd=root,text=True).splitlines()
        for commit in revisions:
            proc=subprocess.run(['git','show',f'{commit}:{relative}'],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            if proc.returncode==0 and hashlib.sha256(proc.stdout).hexdigest()==digest:
                result=dict(sha256=digest,path=relative,git_commit=commit,bytes=len(proc.stdout))
                if not path.exists() or sha(path)!=digest:
                    result['historical_source']=proc.stdout.decode('utf-8');embedded+=len(proc.stdout)
                cache[key]=result;return result
        for backup in base.rglob(path.name):
            if backup.is_file() and backup.stat().st_size<512*2**10 and sha(backup)==digest:
                content=backup.read_bytes();result=dict(sha256=digest,path=relative,preserved_backup=str(backup),bytes=len(content),historical_source=content.decode('utf-8'))
                embedded+=len(content);cache[key]=result;return result
        raise RuntimeError(f'Exact historical source missing: {relative} {digest}')
    for manifest_path in sorted(base.glob('*/manifest.json')):
        manifest=json.loads(manifest_path.read_text());sources=manifest.get('sources',{})
        if not isinstance(sources,dict):continue
        found=[]
        for name,digest in sources.items():
            path=Path(name)
            if path.is_relative_to(root) and path.suffix in ['.py','.sh','.json']:
                item=recover(path,digest);found.append(dict(path=item['path'],sha256=digest))
        receipts.append(dict(manifest=str(manifest_path),manifest_sha256=sha(manifest_path),sources=found))
    assert embedded<8*2**20,'Historical text exceeded bounded source export'
    return dict(complete=True,receipts=receipts,versions=list(cache.values()),embedded_bytes=embedded,
        scope='Exact source bytes for every discovered V14 job manifest. Current execution source is in Git; changed historical versions are embedded and bound to their Git commit/backup.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--cohort',default='formal_001');p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    assert Path(a.cohort).name==a.cohort
    base=a.artifacts.resolve();cohort=base/a.cohort;dest=a.results.resolve()
    assert dest.is_relative_to(Path('/data1/tzh')) and dest.is_dir()
    review=json.loads((cohort/'independent_review.json').read_text());manifest=json.loads((cohort/'manifest.json').read_text());root=Path(manifest['source_root'])
    assert review['complete'] and review['audit_passed'] and review['full_model_qualified'] and all(review['gates'].values())
    assert review['all_original_thresholds_unchanged'] and len(review['comparisons'])==2 and len(review['resources'])==8
    assert source_snapshot(root)==manifest['core_snapshot']
    for sources in [manifest['sources'],review['sources'],review['frozen_sources']]:
        for path,digest in sources.items():assert sha(path)==digest,path
    for path,digest in review['raw_evidence_sha256'].items():assert sha(path)==digest,path
    performance=all(v>=.03 for row in review['comparisons'] for v in row['full_runtime_reductions'].values())
    assert review['performance_rule_passed']==performance and review['performance_adopted']==performance
    mapping={cohort/'independent_review.json':'training_review',cohort/'manifest.json':'full_manifest',cohort/'attempts.json':'full_attempt_history',
        base/'full_runtime_policy_001.json':'qualification_policy',base/'runtime_environment_001.json':'environment',
        base/'full_validation_screen_001/result.json':'validation_screen',base/'full_validation_screen_001/manifest.json':'validation_screen_manifest',
        base/'full_validation_screen_001/attempts.json':'validation_screen_attempts',base/'full_validation_screen_001/CPU_reaudit_001.json':'validation_CPU_audit_correction'}
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or folder==cohort:continue
        for filename in ['manifest.json','attempts.json','acceptance_failure_review_001.json','controller_retired_for_source_fix.json']:
            path=folder/filename
            if path.is_file():mapping.setdefault(path,folder.name+'_'+path.stem)
    for path in sorted(base.glob('validation_compile*.json')):mapping[path]=path.stem
    for path in sorted(base.glob('reviewer_preflight*.json')):mapping[path]=path.stem
    copies=[]
    for source,name in mapping.items():
        assert source.stat().st_size<2*2**20,source
        target=dest/f'qwen3_triton_runtime_v14_{name}_v1.json'
        if target.exists():assert sha(target)==sha(source),target
    # Validate source recovery and all destination paths before exporting.
    historical=historical_sources(root,base)
    special={name:dest/f'qwen3_triton_runtime_v14_{name}_v1.json' for name in ['historical_sources','strategy','evidence_index']}
    role_copies=[]
    for role in manifest['roles']:
        source=cohort/role['name']/'run/result.json';d=json.loads(source.read_text())
        assert d['complete'] and not d['screen_only'] and source.stat().st_size<2*2**20
        target=dest/f'qwen3_triton_runtime_v14_{role["name"]}_raw_v1.json'
        if target.exists():assert sha(target)==sha(source),target
        role_copies.append((source,target))
    for source,name in mapping.items():
        target=dest/f'qwen3_triton_runtime_v14_{name}_v1.json'
        if not target.exists():
            with target.open('xb') as f:f.write(source.read_bytes())
        assert sha(target)==sha(source)
        copies.append(dict(path=target.name,original_path=str(source),sha256=sha(target),bytes=target.stat().st_size))
    # The eight small raw reports preserve all399 recorded error rows. Dense
    # gradients/traces are still excluded; do not imply independent replay.
    for source,target in role_copies:
        if not target.exists():
            with target.open('xb') as f:f.write(source.read_bytes())
        copies.append(dict(path=target.name,original_path=str(source),sha256=sha(target),bytes=target.stat().st_size))
    write_new(special['historical_sources'],historical)
    strategy=dict(complete=True,full_model_qualified=True,performance_adopted=performance,
        entry='scripts/train_qwen3_runtime_v14.py',default_runtime='full',historical_entries_preserved=True,
        model='Qwen/Qwen3-8B',layers=36,world_size=4,local_batch=1,sequence=512,validation_batch_per_rank=4,
        quality_updates=16,total_actual_updates=28,validation_blocks=512,validation_points=[0,16],
        compute='BF16',master_gradient_moments='FP32',deterministic_algorithms=True,deterministic_fill=True,TF32=False,
        GPU_data_path='Triton model forward/backward, FSDP packing/copy/cast/init/accumulation, AdamW, all GPU exchanges/reductions/barriers and validation statistics.',
        host_services='PyTorch FSDP2/autograd CPU state machine, CUDA driver allocation/IPC mappings/streams/events, CPU IO/data preparation and host-device DMA.',
        effective_cap_GiB=41.875,Torch_cap_GiB=41.375,external_peer_reserve_bytes=512*2**20,
        comparisons=review['comparisons'],review_sha256=sha(cohort/'independent_review.json'),input_identity=manifest['input_identity'],
        limitations='Only qualified4A6000/versions,B1S512 and fullvalidationB4. No inference, pretraining convergence or cross-hardware parity claim. Feature qualification is separate from speed adoption; no native GPU fallback in full mode.')
    write_new(special['strategy'],strategy)
    existing=[dict(path=x.name,sha256=sha(x),bytes=x.stat().st_size) for x in sorted(dest.glob('qwen3_triton_runtime_v14_*.json')) if x.name not in {r['path'] for r in copies} and x not in special.values()]
    index=dict(complete=True,full_model_qualified=True,performance_adopted=performance,copies=copies,existing_summaries=existing,
        strategy=dict(path=special['strategy'].name,sha256=sha(special['strategy'])),
        historical_sources=dict(path=special['historical_sources'].name,sha256=sha(special['historical_sources'])),
        raw_evidence_sha256=review['raw_evidence_sha256'],reference_archive_sha256=review['reference_archive_sha256'],
        frozen_sources=manifest['sources'],collector_sha256=sha(Path(__file__)),scope=__doc__)
    write_new(special['evidence_index'],index)
    print(json.dumps(dict(complete=True,full_model_qualified=True,performance_adopted=performance,copied=len(copies),historical_versions=len(historical['versions']))),flush=True)


if __name__=='__main__':main()
