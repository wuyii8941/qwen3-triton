"""Bundle audited V11 results and explicit strategy flags, without raw tensors."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    base,destination=args.artifacts.resolve(),args.results.resolve()
    assert destination.is_dir() and not destination.is_relative_to(Path('/home/tzh'))
    formal=base/'formal_001'
    # Require the completed unified review before creating any public output.
    review=json.loads((formal/'independent_review.json').read_text())
    manifest=json.loads((formal/'manifest.json').read_text())
    assert review['complete'] and review['audit_passed'] and len(manifest['roles'])==6
    for record in (review,manifest):
        for path,digest in record['sources'].items():assert sha(path)==digest,path
    assert sha(formal/'selection_manifest.json')==manifest['selection_manifest_sha256']
    assert sha(formal/'native_feasibility.json')==manifest['native_feasibility_sha256']
    evidence={}
    for role in manifest['roles']:
        name=role['name'];path=formal/name/'result.json';a=json.loads(path.read_text())
        monitor_path=formal/(name+'.gpu.json');monitor=json.loads(monitor_path.read_text())
        assert a['complete'] and not a['screen_only'] and len(a['ranks'])==4
        assert a['model']=='Qwen/Qwen3-8B' and a['layers']==36 and a['world_size']==4
        assert a['input_manifest_sha256']==manifest['input_identity']
        assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
        assert monitor['samples'];owned=set(monitor['known_owned_pids'])
        assert all(not s.get('foreign_pids') and 'observation_error' not in s
            and all(x['pid'] in owned for x in s['apps']) for s in monitor['samples'])
        samples=[r['wall_seconds'] for r in a['ranks'][0]['rows'] if r['measured']]
        assert len(samples)==8 and review['timing'][name]['median_seconds']==statistics.median(samples)
        for rank in a['ranks']:
            assert rank['updates']==27 and rank['parameter_count']==rank['optimizer_state_count']==399
            assert rank['all_master_grad_moment_finite_fp32']
            assert max(rank['measured_training_peak'].values())<=41.875
            assert samples==[r['wall_seconds'] for r in rank['rows'] if r['measured']]
        trace=formal/name/'training_trace.json'
        assert sha(trace)==review['traces'][name]['trace_sha256']
        for p in (path,monitor_path,trace):evidence[str(p)]=sha(p)
    for order in (1,2):
        c=review['comparisons'][order-1]
        baseline=review['timing'][f'baseline_r{order}']['median_seconds']
        candidate=review['timing'][f'candidate_r{order}']['median_seconds']
        hf=review['timing'][f'hf_r{order}']['median_seconds']
        assert c['order']==order and c['reduction_vs_baseline']==1-candidate/baseline
        assert c['reduction_vs_HF_same_strategy']==1-candidate/hf
    numerical=all(review['gates'].values())
    performance=all(c['reduction_vs_baseline']>=manifest['minimum_reduction_each_order'] for c in review['comparisons'])
    assert review['all_numerical_quality_CUDA_gates_passed']==numerical
    assert review['performance_rule_passed']==performance
    assert review['qualified_training_improvement']==(numerical and performance)
    mapping={
        'formal_001/independent_review.json':'training_review',
        'formal_001/manifest.json':'cohort_manifest',
        'formal_001/selection_manifest.json':'selection_policy',
        'formal_001/native_feasibility.json':'native_feasibility',
        'strategy_screen_002/result.json':'strategy_screen',
        'strategy_screen_002/manifest.json':'strategy_manifest',
        'retention_screen_001/result.json':'retention_screen',
        'retention_screen_001/manifest.json':'retention_manifest',
        'higher_retention_profiles_001.json':'retention_profiles',
        'CPU_preflight_002.json':'trainer_source_checks',
        'CPU_trace_contract_001.json':'trace_contract_checks',
        'CPU_pipeline_preflight_001.json':'pipeline_checks',
        'communication_model_001.json':'communication_model',
        'pre_admission_source_fix_001.json':'startup_disclosure',
    }
    paths={source:destination/f'qwen3_triton_v11_{name}_v1.json' for source,name in mapping.items()}
    strategy_path=destination/'qwen3_triton_v11_strategy_v1.json'
    index_path=destination/'qwen3_triton_v11_evidence_index_v1.json'
    assert all(not p.exists() for p in [*paths.values(),strategy_path,index_path])
    # Validate every source before copying, retaining screening-only flags verbatim.
    for source in paths:json.loads((base/source).read_text())
    for folder in ['strategy_screen_002','retention_screen_001']:
        screen=json.loads((base/folder/'result.json').read_text())
        assert screen['complete'] and not screen['full_gradient_quality_qualified']
        for row in screen['rows']:
            name=row['role']['name'];monitor_path=base/folder/(name+'.gpu.json')
            monitor=json.loads(monitor_path.read_text())
            assert not monitor['foreign_process_seen'] and not monitor['observation_errors']
            if row['completed']:
                assert monitor['returncode']==0 and sha(base/folder/name/'result.json')==row['result_sha256']
            else:
                assert monitor['returncode'] and 'out of memory' in (base/folder/(name+'.log')).read_text().lower()
            evidence[str(monitor_path)]=sha(monitor_path)
    copied=[]
    for source,path in paths.items():
        raw=(base/source).read_bytes()
        with path.open('xb') as stream:stream.write(raw)
        assert sha(path)==sha(base/source)
        copied.append(dict(path=path.name,original_path=str(base/source),sha256=sha(path),bytes=len(raw)))
    selected=manifest['selected_strategy']
    flags=['--keep-full-layers',str(selected['keep_full_layers']),'--checkpoint-every',str(selected['checkpoint_every'])]
    if selected['skip_embedding_backward_gather']:flags.append('--skip-embedding-backward-gather')
    if selected['retain_lm_head']:flags.append('--retain-lm-head')
    triton_flags=flags+(['--gemm-plan','results/qwen3_triton_v10_gemm_plan_001.json'] if selected['gemm_plan'] else [])
    strategy=dict(complete=True,qualified_training_improvement=review['qualified_training_improvement'],default_changed=False,
        model='Qwen/Qwen3-8B',layers=36,world_size=4,batch_per_rank=1,sequence_length=512,
        hardware='RTX A6000 SM86',memory_cap_GiB=41.875,compute='BF16',master_gradient_moments='FP32',
        input_identity=manifest['input_identity'],strategy=selected,
        entry='scripts/train_qwen3_training_v11.py',extra_flags={'hf':flags,'triton_optimized':triton_flags},
        comparisons=review['comparisons'],review_sha256=sha(formal/'independent_review.json'),
        scope='Explicit measured strategy; this JSON is a descriptor, not a new CLI loader. Supply the listed flags and original input/quality arguments. No inference or convergence claim.')
    with strategy_path.open('x') as stream:stream.write(json.dumps(strategy,indent=2)+'\n')
    index=dict(complete=True,qualified_training_improvement=review['qualified_training_improvement'],default_changed=False,
        copies=copied,strategy=dict(path=strategy_path.name,sha256=sha(strategy_path)),
        raw_evidence_sha256=evidence,collector_sha256=sha(Path(__file__)),
        scope='Byte-identical small evidence copies plus an explicit strategy descriptor. Models, corpus, caches, raw gradient archives, logs and profiler traces are not included. Original paths/hashes support local reinspection.',
        limitations=review['limitation'])
    with index_path.open('x') as stream:stream.write(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(complete=True,copied_files=len(copied),qualified_training_improvement=review['qualified_training_improvement'],strategy=str(strategy_path)),indent=2))


if __name__=='__main__':main()
