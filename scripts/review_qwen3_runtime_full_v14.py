"""Independent CPU reference/archive, resource, numerical, memory and trace audit.

Dense candidate gradients are compared in the actual run and are not archived.
This audit replays the original reference hashes/norms and recorded error
arithmetic; it does not claim a second independent candidate-gradient replay.
Functionality qualification and matched-order speed recommendation are separate.
"""
import argparse,json,statistics
from pathlib import Path
import torch
from qwen3_triton.gradient_archive import load,sha
from training.provenance import source_snapshot
from qwen3_runtime_v14_review import ARMS,run_review,trace_review


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert out.is_relative_to(Path('/data1/tzh'))
    manifest=json.loads((out/'manifest.json').read_text());root=Path(manifest['source_root'])
    assert source_snapshot(root)==manifest['core_snapshot']
    for path,digest in manifest['sources'].items():assert sha(path)==digest,path
    policy=json.loads(Path(manifest['policy']).read_text());assert sha(Path(manifest['policy']))==manifest['policy_sha256']
    assert (manifest['quality_steps'],manifest['warmup'],manifest['measured'],manifest['total_updates'])==(16,3,8,28)
    assert manifest['effective_cap_GiB']==41.875 and manifest['minimum_reduction_each_order']==.03
    expected=[f'{arm}_r{order}' for order,key in [(1,'first_order'),(2,'second_order')] for arm in policy[key]]
    roles=manifest['roles'];assert [r['name'] for r in roles]==expected
    assert policy['first_order']==list(ARMS) and policy['second_order']==list(reversed(ARMS))
    reference=Path(manifest['reference_dir']);ref=json.loads((reference/'result.json').read_text())
    assert sha(reference/'result.json')==manifest['reference_result_sha256']
    assert ref['variant']=='hf' and ref['layers']==36 and ref['input_manifest_sha256']==manifest['input_identity']
    assert ref['math_policy']['allow_bf16_reduced_precision_reduction'] is False
    torch.set_num_threads(2);canonical=[];norms=[];archive_hashes={}
    for rank in range(4):
        path=reference/f'raw_rank{rank}.pt.gz';digest=sha(path)
        assert digest==ref['ranks'][rank]['reference_archive']['sha256'];archive_hashes[str(path)]=digest
        raw=load(path)['1'];assert raw['ids'].shape==(1,512) and raw['logits'].shape==(1,512,151936)
        assert len(raw['gradients'])==399 and all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in raw['gradients'].values())
        canonical.append(set(raw['gradients']));norms.append({name:float(t.norm()) for name,t in raw['gradients'].items()});del raw
    history=json.loads((out/'attempts.json').read_text());assert [r['number'] for r in history]==list(range(1,len(history)+1))
    assert [r['role'] for r in history if r['accepted']]==expected
    plan=json.loads((root/'results/qwen3_triton_v10_gemm_plan_001.json').read_text())
    screen=Path(manifest['validation_screen']);sm=json.loads((screen/'manifest.json').read_text())
    for path,digest in sm['sources'].items():
        source=Path(path)
        if source.is_relative_to(root/'qwen3_triton') or source.is_relative_to(root/'training') or source in [
            root/'scripts/train_qwen3_runtime_v14.py',root/'scripts/train_qwen3_vocab_v12.py',root/'scripts/vocab_model_v12.py',
            root/'scripts/research_python.sh',root/'configs/fsdp_8b.json',root/'results/qwen3_triton_v10_gemm_plan_001.json',
            Path(manifest['input_root'])/'artifacts/official_qwen3_inputs_v1.json']:
            assert sha(source)==digest,source
    screened=json.loads((screen/'accepted/run/result.json').read_text())
    screen_checks=run_review(screened,dict(arm='full_runtime'),formal=False,reference_sha=manifest['reference_result_sha256'],canonical=canonical,norms=norms)
    screen_traces=trace_review(screen/'accepted/run','full',plan,validation=True)
    assert manifest['prerequisite_reaudit']==dict(execution_sources_current=True,checks=screen_checks,traces=screen_traces)
    assert all(screen_checks['gates'].values()) and all(t['passed'] for t in screen_traces.values())
    checks={};traces={};data={};resources=[];evidence={}
    for role in roles:
        name=role['name'];attempts=[r for r in history if r['role']==name];clean=[r for r in attempts if not r['resource_contaminated']]
        assert len(clean)==1 and clean[0] is attempts[-1] and clean[0]['accepted'] and clean[0]['returncode']==0
        assert (out/name).resolve()==(out/clean[0]['folder']).resolve()
        for row in attempts:
            folder=out/row['folder']
            for filename,key in [('monitor.json','monitor_sha256'),('worker.log','log_sha256')]:
                assert sha(folder/filename)==row[key];evidence[str(folder/filename)]=row[key]
        monitor=json.loads((out/name/'monitor.json').read_text());owned=set(monitor['known_owned_pids'])
        assert monitor['returncode']==0 and monitor['samples'] and not monitor['foreign_process_seen'] and not monitor['observation_errors']
        assert all(not s.get('foreign_pids') and 'observation_error' not in s and all(p['pid'] in owned for p in s['apps']) for s in monitor['samples'])
        resources.append(dict(role=name,attempt=clean[0]['number'],monitor_sha256=sha(out/name/'monitor.json')))
        folder=out/name/'run';path=folder/'result.json';assert sha(path)==clean[0]['result_sha256'];d=json.loads(path.read_text())
        assert d['input_manifest_sha256']==manifest['input_identity'] and d['input_root']==manifest['input_root']
        for source,digest in d['sources'].items():assert sha(source)==digest,source
        checks[name]=run_review(d,role,formal=True,reference_sha=manifest['reference_result_sha256'],canonical=canonical,norms=norms)
        runtime,_=ARMS[role['arm']];traces[name]=trace_review(folder,runtime,plan,validation=True)
        assert checks[name]==clean[0]['checks'] and traces[name]==clean[0]['traces']
        for filename in ['result.json',*d['ranks'][0]['traces']]:evidence[str(folder/filename)]=sha(folder/filename)
        data[name]=d
    quality={};comparisons=[]
    for role in roles:
        name=role['name'];row=data[name]['ranks'][0];base=data[f'HF_native_r{role["order"]}']['ranks'][0]
        drift=[abs(x['global_loss']-y['global_loss']) for x,y in zip(row['rows'][:16],base['rows'][:16])]
        validation=[abs(x['loss']-y['loss']) for x,y in zip(row['full_validation'],base['full_validation'])]
        quality[name]=dict(mean_loss_drift=statistics.mean(drift),max_loss_drift=max(drift),validation_drifts=validation,
            passed=statistics.mean(drift)<.002 and max(drift)<.01 and max(validation)<.002)
    for order in [1,2]:
        times={arm:checks[f'{arm}_r{order}']['median_seconds'] for arm in ARMS}
        comparisons.append(dict(order=order,median_seconds=times,
            full_runtime_reductions={arm:1-times['full_runtime']/times[arm] for arm in ARMS if arm!='full_runtime'},
            TritonAdamW_reduction=1-times['V13_TritonAdamW']/times['V13'],
            remaining_runtime_reduction=1-times['full_runtime']/times['V13_TritonAdamW']))
    gates={key:all(c['gates'][key] for c in checks.values()) for key in ['numerics','effective_memory','lossless_transport']}
    gates.update(short_quality=all(q['passed'] for q in quality.values()),actual_phase_GPU_coverage=all(t['passed'] for rows in traces.values() for t in rows.values()),
        original_reference_archives=True,source_identity=True,first_clean_resource_selection=True)
    performance=all(g>=.03 for row in comparisons for g in row['full_runtime_reductions'].values())
    report=dict(complete=True,audit_passed=True,full_model_qualified=all(gates.values()),performance_rule_passed=performance,
        performance_adopted=all(gates.values()) and performance,gates=gates,checks=checks,traces=traces,quality=quality,comparisons=comparisons,
        reference_archive_sha256=archive_hashes,raw_evidence_sha256=evidence,resources=resources,all_original_thresholds_unchanged=True,
        sources={str(p):sha(p) for p in [Path(__file__),root/'scripts/qwen3_runtime_v14_review.py',out/'manifest.json',out/'attempts.json']},
        frozen_sources=manifest['sources'],scope=__doc__,
        limitations='FourA6000,B1S512,16quality updates, not convergence/statistical significance/inference.28actual updates including one untimed real profile update. Full initialization/first-update/complete-step/full-validation traces only full_runtime are required entirely Triton. Controls intentionally contain native runtime kernels. CPU orchestration, allocator/mapping/driver/events and host-device DMA remain host services. Effective memory uses512MiB external reserve, not NVML total. No main merge/default performance recommendation implied.')
    with (out/'independent_review.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['complete','full_model_qualified','performance_adopted','gates','comparisons']},indent=2))


if __name__=='__main__':main()
