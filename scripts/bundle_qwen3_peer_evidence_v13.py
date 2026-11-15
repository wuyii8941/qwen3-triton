"""Publish only qualified V13 summaries, never raw weights/gradients/traces.

Refuses incomplete or failed whole-cohort reviews and source drift. Historical
negative screens retain their own source archives; only the qualified peer
method can produce a recommended strategy. Original defaults stay unchanged.
"""
import argparse,json
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from training.provenance import source_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    b=a.artifacts.resolve();destination=a.results.resolve();formal=b/'formal_001'
    assert destination.is_relative_to(Path('/data1/tzh')) and destination.is_dir()
    review=json.loads((formal/'independent_review.json').read_text())
    assert review['complete'] and review['audit_passed'] and review['qualified_training_improvement']
    assert all(review['gates'].values()) and review['performance_rule_passed'] and review['all_original_thresholds_unchanged']
    manifest=json.loads((formal/'manifest.json').read_text());root=Path(manifest['source_root'])
    for sources in [manifest['sources'],review['sources'],review['frozen_sources']]:
        for path,digest in sources.items():assert sha(path)==digest,path
    assert source_snapshot(root)==manifest['core_snapshot']
    for path,digest in review['raw_evidence_sha256'].items():assert sha(path)==digest,path
    assert len(review['resource_selection'])==8 and len(review['comparisons'])==2
    assert all(g>=.03 for c in review['comparisons'] for g in c['candidate_reductions'].values())
    orchestration=json.loads((b/'review_orchestration_001.json').read_text())
    assert orchestration['stage']=='CPU_review_complete' and orchestration['qualified_training_improvement']
    assert orchestration['review_sha256']==sha(formal/'independent_review.json')
    assert orchestration['reviewer_sha256']==sha(root/'scripts/review_qwen3_peer_full_v13.py')
    # Preserve exact historical source identities, even where the worktree
    # subsequently added metadata hooks or strengthened the wire check.
    archives=['micro_001_sources','full_screen_001_sources','probe_001_sources','symmetric_rs_micro_002_sources']
    for folder in archives:
        archive=json.loads((b/folder/'receipt.json').read_text())
        assert archive['complete']
        for row in archive['sources']:
            assert sha(row.get('archived',row.get('archived_path')))==row['sha256']
    mapping={
        'formal_001/independent_review.json':'training_review',
        'formal_001/manifest.json':'full_manifest',
        'formal_001/attempts.json':'full_attempt_history',
        'peer_pilot_001/result.json':'peer_pilot',
        'peer_pilot_001/manifest.json':'peer_pilot_manifest',
        'HF_peer_pilot_001/result.json':'HF_peer_pilot',
        'HF_peer_pilot_001/manifest.json':'HF_peer_pilot_manifest',
        'symmetric_probe_001/probe.json':'probe_API_failure',
        'probe_001_sources/receipt.json':'probe_API_failure_source_archive',
        'symmetric_probe_002/probe.json':'symmetric_probe',
        'symmetric_probe_002/manifest.json':'symmetric_probe_manifest',
        'full_screen_001/manifest.json':'short_screen_manifest',
        'full_screen_001/attempts.json':'short_screen_attempts',
        'full_screen_001_sources/receipt.json':'short_screen_source_archive',
        'micro_001_sources/receipt.json':'communication_micro_source_archive',
        'symmetric_rs_micro_002_sources/receipt.json':'symmetric_micro_source_archive',
        'symmetric_rs_micro_002/manifest.json':'symmetric_micro_manifest',
        'CPU_peer_preflight_001.json':'CPU_peer_checks',
        'adoption_policy_001.json':'adoption_policy',
        'review_orchestration_001.json':'review_orchestration',
    }
    paths={src:destination/f'qwen3_triton_v13_{name}_v1.json' for src,name in mapping.items()}
    strategy_path=destination/'qwen3_triton_v13_strategy_v1.json';index_path=destination/'qwen3_triton_v13_evidence_index_v1.json'
    assert all(not p.exists() for p in [*paths.values(),strategy_path,index_path])
    for src in paths:json.loads((b/src).read_text());assert (b/src).stat().st_size<2**20
    # Existing small negative/module summaries are separately bound to their
    # raw evidence, not replaced with a fabricated success-only history.
    existing=['communication_micro_001','protocol_screen_001','short_screen_review_v1','symmetric_rs_micro_v1']
    existing_paths=[destination/f'qwen3_triton_v13_{name}.json' for name in existing]
    assert all(p.exists() for p in existing_paths)
    assert json.loads(existing_paths[2].read_text())['selected'] is None
    for role in manifest['roles']:
        path=formal/role['name']/'run/result.json';d=json.loads(path.read_text())
        assert d['complete'] and not d['screen_only']
        for rank in d['ranks']:
            assert rank['updates']==27 and rank['parameter_count']==rank['optimizer_state_count']==399
            assert rank['all_master_grad_moment_finite_fp32'] and max(rank['effective_training_peak'].values())<=41.875
            if role['peer']:assert rank['communication']['lossless'] and rank['communication']['lossless_violation_flag']==0
    strategy=dict(complete=True,qualified_training_improvement=True,default_changed=False,
        entry='scripts/train_qwen3_peer_rs_v13.py',model='Qwen/Qwen3-8B',layers=36,world_size=4,batch_per_rank=1,sequence=512,
        compute='BF16',master_gradient_moments='FP32',deterministic_fill=True,
        gradient_transport='Every staged FP32 coordinate bit-checked for exact BF16 roundtrip; FP32 accumulation/output; other sizes native FP32 NCCL.',
        effective_memory_cap_GiB=41.875,torch_allocator_cap_GiB=41.375,external_memory_reserve_bytes=512*2**20,
        extra_flags=['--variant','triton_optimized','--vocab-ce','--fused-vocab-dw','--keep-full-layers','36','--checkpoint-every','1',
            '--skip-embedding-backward-gather','--gemm-plan','results/qwen3_triton_v10_gemm_plan_001.json'],
        comparisons=review['comparisons'],measured_peaks={name:value for name,value in review['timing'].items() if name.startswith('candidate')},
        qualification_review_sha256=sha(formal/'independent_review.json'),input_identity=manifest['input_identity'],
        scope='Descriptor, not CLI loader. Add original input/full-quality/reference/output arguments. Fixed Torch2.10 CUDA-backend alpha API, four RTX A6000, B1/S512. No inference/convergence claim. HF can use the same custom Comm; this is not an exclusive algorithm.')
    copies=[]
    for src,target in paths.items():
        payload=(b/src).read_bytes()
        with target.open('xb') as f:f.write(payload)
        assert sha(target)==sha(b/src)
        copies.append(dict(path=target.name,original_path=str(b/src),sha256=sha(target),bytes=len(payload)))
    with strategy_path.open('x') as f:f.write(json.dumps(strategy,indent=2)+'\n')
    index=dict(complete=True,qualified_training_improvement=True,copies=copies,
        strategy=dict(path=strategy_path.name,sha256=sha(strategy_path)),collector_sha256=sha(Path(__file__)),
        existing_summaries=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in existing_paths],
        raw_evidence_sha256=review['raw_evidence_sha256'],reference_archive_sha256=review['reference_archive_sha256'],
        frozen_sources=manifest['sources'],scope='Small JSON evidence and source only. Raw traces/logs/gradient archives/weights/data/environments remain local. Failed probe is an API invocation error, not hardware unavailability. Negative screens are preserved and not recommended.')
    with index_path.open('x') as f:f.write(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(complete=True,copied=len(copies),comparisons=review['comparisons']),indent=2))


if __name__=='__main__':main()
