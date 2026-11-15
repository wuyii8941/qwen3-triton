"""CPU-only correction of an audit bug; reuse the first clean GPU result.

Never reruns/selects timings, changes model execution, alters tolerances, or
discards the prior rejection. Preserves original history/state and records the
exact extra finalnorm prefetch observed after no-grad FSDP validation.
"""
import argparse,collections,gzip,json
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_runtime_v14_review import run_review,trace_review
from run_vocab_resource_queue_v12 import dump


def annotation_counts(path):
    events=json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
    return dict(collections.Counter(e['name'] for e in events if e.get('cat')=='user_annotation' and
        (e.get('name','').startswith('FSDP::all_gather (') or e.get('name')=='FSDP::backward_prefetch for model.norm')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert out.is_relative_to(Path('/data1/tzh')) and not (out/'result.json').exists()
    manifest=json.loads((out/'manifest.json').read_text());root=Path(manifest['source_root'])
    old_state=json.loads((out/'state.json').read_text());assert old_state['stage']=='clean_technical_failure'
    assert not Path(f'/proc/{old_state["pid"]}').exists(),'Review only after original controller is terminal'
    history=json.loads((out/'attempts.json').read_text());clean=[r for r in history if not r['resource_contaminated']]
    assert len(clean)==1 and clean[0] is history[-1] and clean[0]['returncode']==0 and not clean[0]['accepted']
    row=clean[0];folder=out/row['folder'];run=folder/'run';monitor=json.loads((folder/'monitor.json').read_text())
    assert sha(folder/'monitor.json')==row['monitor_sha256'] and sha(folder/'worker.log')==row['log_sha256']
    assert monitor['returncode']==0 and not monitor['foreign_process_seen'] and not monitor['observation_errors']
    for source,digest in manifest['sources'].items():
        if Path(source)==root/'scripts/qwen3_runtime_v14_review.py':continue  # the explicitly documented CPU audit correction
        assert sha(source)==digest,source
    d=json.loads((run/'result.json').read_text());assert d['screen_only'] and d['protocol']['total_optimizer_updates']==5
    reference_sha=d['ranks'][0]['gradient_comparisons'][0]['reference_result_sha256']
    checks=run_review(d,dict(arm='full_runtime'),formal=False,reference_sha=reference_sha)
    plan=json.loads((root/'results/qwen3_triton_v10_gemm_plan_001.json').read_text());traces=trace_review(run,'full',plan,validation=True)
    assert all(checks['gates'].values()) and all(t['passed'] for t in traces.values())
    before=annotation_counts(Path(manifest['pilot'])/'accepted/run/initial_step_trace.json.gz')
    after=annotation_counts(run/'initial_step_trace.json.gz')
    assert before['FSDP::all_gather (model.norm)']==2 and after['FSDP::all_gather (model.norm)']==3
    assert before.get('FSDP::backward_prefetch for model.norm',0)==0 and after['FSDP::backward_prefetch for model.norm']==1
    for key,count in before.items():
        if key!='FSDP::all_gather (model.norm)':assert after[key]==count,key
    torch_sources=[Path('/data1/tzh/envs/ka_main/lib/python3.11/site-packages/torch/distributed/fsdp/_fully_shard')/x for x in ['_fsdp_param_group.py','_fsdp_state.py']]
    report=dict(complete=True,CPU_only=True,no_GPU_rerun=True,full_model_qualified=False,original_state=old_state,
        original_attempt_history_sha256=sha(out/'attempts.json'),source_manifest_sha256=sha(out/'manifest.json'),raw_result_sha256=sha(run/'result.json'),
        correction='FSDP retains no-grad post_forward_order until backward cleanup. Each validation adds100forwardAG plus1mistargeted finalnorm prefetch in the next training backward:202AG across two validations, not200.',
        numerical_quality_memory_thresholds_unchanged=True,first_clean_attempt_preserved=True,before_module_annotations=before,after_module_annotations=after,
        checks=checks,traces=traces,sources={str(x):sha(x) for x in [Path(__file__),root/'scripts/qwen3_runtime_v14_review.py',root/'scripts/qwen3_runtime_v14_contract.py',*torch_sources]},
        scope=__doc__)
    receipt=out/'CPU_reaudit_001.json'
    with receipt.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    (out/'attempts_before_CPU_reaudit_001.json').write_bytes((out/'attempts.json').read_bytes())
    row.update(accepted=True,accepted_by_CPU_reaudit=True,result_sha256=sha(run/'result.json'),CPU_reaudit_sha256=sha(receipt))
    dump(out/'attempts.json',history);(out/'accepted').symlink_to(folder.relative_to(out),target_is_directory=True)
    dump(out/'result.json',dict(complete=True,eligible_for_formal=True,full_model_qualified=False,checks=checks,traces=traces,CPU_reaudit=str(receipt),
        scope='First clean5update actual8B result,512blocks at0/1,B4. Corrected CPU count audit only;16quality formal cohort remains pending.'))
    dump(out/'state.json',dict(pid=old_state['pid'],stage='complete_validation_feasibility_only',CPU_reaudit=str(receipt),no_GPU_rerun=True))
    print(json.dumps(dict(complete=True,CPU_only=True,no_GPU_rerun=True,gates=checks['gates'],all_traces_passed=True)),flush=True)


if __name__=='__main__':main()
