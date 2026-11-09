import json
import hashlib
from pathlib import Path
import pytest
from scripts.report_fsdp_s2048_confirmation import audit
from scripts.confirm_fsdp_s2048 import decisions
from scripts.summarize_fsdp_shape_confirmation import summarize
from tests.test_fsdp_memory_summary import result


def setup(root):
    policies=dict(base=(8,2,'full',0),native=(32,1,'full',0),ce=(12,2,'full',128))
    initial=result()
    workload={k:initial[k] for k in ('model','world_size','batch','seq','warmup','steps',
        'memory_budget_gib','precision','attention')};workload.update(seq=2048,steps=12)
    spec=dict(policies=policies,candidate='native',workload=workload)
    selection=dict(baseline='base',no_ce='native',ce='ce')
    order=['r0_base','r0_native','r0_ce','r1_ce','r1_native','r1_base']
    for name in order:
        policy=name[3:];keep,every,mode,chunk=policies[policy];seconds=dict(base=8.,native=6.,ce=7.)[policy]
        r=result();r.update(workload,median_step_seconds=seconds,global_tokens_per_second=8192/seconds)
        r['records']=[dict(step=i+1,warmup=i<3,max_rank_seconds=seconds,loss=2.,
            peak_allocated_gib=39.,peak_reserved_gib=42.) for i in range(15)]
        r['optimization'].update(keep_full_layers=keep,checkpoint_every=every,checkpoint_mode=mode,ce_only_chunk_size=chunk)
        folder=root/name;folder.mkdir();(folder/'result.json').write_text(json.dumps(r))
        (root/(name+'.log')).write_text('completed')
    (root/'status.log').write_text('\n'.join(n+' returncode=0' for n in order)+'\n')
    for name,(keep,every,mode,chunk) in policies.items():
        rows=[dict(passed=True,tied=t,backend='official',dtype='torch.bfloat16',world=4,head_dim=128,
            seq=2048,steps=3,deterministic=True,checkpoint_every=every,checkpoint_mode=mode,
            keep_full_layers=1,loss_chunk_size=0,ce_only_chunk_size=chunk,swiglu=False,
            accumulation=1,gradient_sync='each') for t in (False,True)]
        (root/f'gate_{name}.log').write_text('\n'.join(json.dumps(r) for r in rows))
    report=summarize(root,spec);decision=decisions(report,selection)
    report.update(decision=decision,reference_content_sha256='frozen',
        supervisor_source_sha256=hashlib.sha256(Path('scripts/confirm_fsdp_s2048.py').read_bytes()).hexdigest())
    for name,value in (('result',report),('spec',spec),('selection',selection),('status',dict(phase='finished',decision=decision))):
        (root/(name+'.json')).write_text(json.dumps(value))
    return dict(complete=True,content_sha256='frozen')


def test_rebuild_report_from_raw_paired_results_and_gates(tmp_path):
    r=audit(tmp_path,setup(tmp_path))
    assert r['decision']['candidate_for_quality_validation']=='native'
    assert len(r['numerical_gates'])==3


@pytest.mark.parametrize('fault',['status','report','decision','reference','source','gate'])
def test_reject_inconsistent_final_evidence(tmp_path,fault):
    reference=setup(tmp_path)
    if fault=='status':
        path=tmp_path/'status.json';r=json.loads(path.read_text());r['phase']='running'
    elif fault=='gate':
        path=tmp_path/'gate_ce.log';r=json.loads(path.read_text().splitlines()[0]);r['passed']=False
        path.write_text(json.dumps(r)+'\n')
    elif fault=='reference':reference['content_sha256']='different'
    else:
        path=tmp_path/'result.json';r=json.loads(path.read_text())
        if fault=='report':r['records'][0]['median_step_seconds']=1.
        elif fault=='decision':r['decision']['defaults_changed']=True
        else:r['supervisor_source_sha256']='changed'
    if fault not in ('gate','reference'):path.write_text(json.dumps(r))
    with pytest.raises(ValueError):audit(tmp_path,reference)
