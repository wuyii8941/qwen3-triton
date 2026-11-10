import pytest
import hashlib
import json
from scripts.audit_fsdp_exchange_confirmation import callback_contract,loss_comparison
from scripts.confirm_fsdp_exchange import CASES
from scripts.audit_fsdp_exchange_confirmation import audit,audit_launch
from scripts.confirm_fsdp_exchange import decisions


def test_all_decoder_units_on_all_ranks_must_execute_exactly():
    result={'exchange_callback_counts_by_rank':[
        {f'model.layers.{i}':15 for i in range(36)} for _ in range(4)]}
    callback_contract(result,CASES['torch_decoder_auto_k32'])
    result['exchange_callback_counts_by_rank'][2]['model.layers.0']=0
    with pytest.raises(ValueError):callback_contract(result,CASES['torch_decoder_auto_k32'])


def test_head_and_native_contracts_do_not_accept_other_units():
    result={'exchange_callback_counts_by_rank':[{'lm_head':15} for _ in range(4)]}
    callback_contract(result,CASES['triton_head_simple_k32'])
    with pytest.raises(ValueError):callback_contract(result,CASES['native_auto_k36'])
    result['exchange_callback_counts_by_rank']=[{} for _ in range(4)]
    callback_contract(result,CASES['native_auto_k36'])


def test_loss_tolerance_reports_but_does_not_claim_convergence():
    baseline={'records':[{'loss':2.},{'loss':1.9}]}
    near={'records':[{'loss':2.000001},{'loss':1.900001}]}
    assert loss_comparison(near,baseline)['within_declared_loss_tolerance']
    near['records'][1]['loss']=1.91
    assert not loss_comparison(near,baseline)['within_declared_loss_tolerance']
    near['records'][1]['loss']=float('nan')
    assert not loss_comparison(near,baseline)['within_declared_loss_tolerance']


def test_loss_history_cannot_be_truncated():
    with pytest.raises(ValueError):loss_comparison({'records':[]},{'records':[{'loss':2.}]})


def synthetic_campaign(root):
    """Synthetic records test the auditor, not model correctness/performance."""
    rows=[];sources={'sha256':'a'*64};helpers={'benchmarks/adapter.py':'b'*64}
    times=dict(zip(CASES,(6.,5.,6.5,5.5)))
    for repeat,order in ((0,list(CASES)),(1,list(reversed(CASES)))):
        for name in order:
            keep,backend,scope,proto=CASES[name];label=f'r{repeat}_{name}'
            path=root/label;path.mkdir()
            log=root/(label+'.log');log.write_text('synthetic successful worker\n')
            seconds=times[name]
            records=[dict(step=i+1,warmup=i<3,max_rank_seconds=seconds,loss=2.,
                peak_allocated_gib=38.,peak_reserved_gib=39.) for i in range(15)]
            keys=({f'model.layers.{i}' for i in range(36)} if scope=='decoder'
                  else {'lm_head'} if scope=='head' else set()) if backend else set()
            result=dict(complete=True,backend='official',world_size=4,model='synthetic8b',
                global_parameters=8190735360,batch=1,seq=512,warmup=3,steps=12,
                precision='FP32/BF16',attention='Flash',memory_budget_gib=42.,
                median_step_seconds=seconds,global_tokens_per_second=2048/seconds,
                initial_hf_parity={'max_abs':0.},records=records,
                optimization=dict(keep_full_layers=keep,checkpoint_every=1,checkpoint_mode='full',
                    exchange_reduce=backend,exchange_scope=scope),
                communication_environment={k:proto if k=='NCCL_PROTO' else None for k in
                    ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE')},
                exchange_callback_counts_by_rank=[{k:15 for k in keys} for _ in range(4)],
                manifest=dict(implementation_sources=sources,input_sha256='c'*64,
                    input_sha256_by_rank=['c'*64]*4,exchange_sources=helpers,
                    pytorch='synthetic',cuda='synthetic',gpu='synthetic'))
            raw=json.dumps(result).encode();(path/'result.json').write_bytes(raw)
            rows.append(dict(repeat=repeat,case=name,status='passed',returncode=0,
                source=str(path/'result.json'),sha256=hashlib.sha256(raw).hexdigest(),
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                median_step_seconds=seconds,peak_allocated_gib=38.,peak_reserved_gib=39.))
    (root/'result.json').write_text(json.dumps(dict(complete=True,records=rows,
        implementation_sources=sources,helper_sources=helpers,decisions=decisions(rows))))


def test_full_synthetic_raw_file_audit(tmp_path):
    synthetic_campaign(tmp_path)
    result=audit(tmp_path)
    assert result['raw_runs_audited']==8
    assert result['candidates_for_quality']==['torch_decoder_auto_k32','triton_head_simple_k32']
    assert not result['convergence_proven'] and not result['defaults_changed']


@pytest.mark.parametrize('fault',['result_bytes','log_bytes','summary_metric','decision','missing_run'])
def test_full_audit_rejects_corrupt_evidence(tmp_path,fault):
    synthetic_campaign(tmp_path)
    if fault=='result_bytes':
        path=tmp_path/'r0_native_auto_k36/result.json';path.write_bytes(path.read_bytes()+b' ')
    elif fault=='log_bytes':
        (tmp_path/'r0_native_auto_k36.log').write_text('changed log')
    else:
        path=tmp_path/'result.json';result=json.loads(path.read_text())
        if fault=='summary_metric':result['records'][0]['median_step_seconds']=4.
        if fault=='decision':result['decisions']['torch_decoder_auto_k32']['over_two_percent_both_orders']=False
        if fault=='missing_run':result['records'].pop()
        path.write_text(json.dumps(result))
    with pytest.raises(ValueError):audit(tmp_path)


def resources(root,label,*,free=45568,util=0):
    policy=dict(allocator_budget_gib=42.,free_margin_gib=2.5,idle_confirmations=3)
    data=dict(**policy,samples=[dict(unix_seconds=100.+15*i,
        readings=[[3000,free,util] for _ in range(4)]) for i in range(3)])
    (root/(label+'_resources.json')).write_text(json.dumps(data))
    return policy


def test_launch_observations_are_audited_separately_from_model_speed(tmp_path):
    policy=resources(tmp_path,'case')
    assert len(audit_launch(tmp_path,'case',policy))==64
    resources(tmp_path,'case',free=45567)
    with pytest.raises(ValueError,match='guard'):audit_launch(tmp_path,'case',policy)
    resources(tmp_path,'case',util=10)
    with pytest.raises(ValueError,match='guard'):audit_launch(tmp_path,'case',policy)


def test_full_audit_includes_all_prelaunch_resource_hashes(tmp_path):
    synthetic_campaign(tmp_path)
    path=tmp_path/'result.json';summary=json.loads(path.read_text())
    for row in summary['records']:
        label=f'r{row["repeat"]}_{row["case"]}'
        summary['resource_policy']=resources(tmp_path,label)
    path.write_text(json.dumps(summary))
    report=audit(tmp_path)
    assert report['launch_resource_guard_audited'] and len(report['launch_resource_sha256'])==8


def test_faster_native_protocol_control_is_not_discarded(tmp_path):
    synthetic_campaign(tmp_path)
    path=tmp_path/'result.json';summary=json.loads(path.read_text())
    for row in summary['records']:
        if row['case']!='native_simple_k36':continue
        worker=tmp_path/f'r{row["repeat"]}_{row["case"]}'/'result.json'
        report=json.loads(worker.read_text())
        for step in report['records']:step['max_rank_seconds']=5.
        report['median_step_seconds']=5.;report['global_tokens_per_second']=2048/5
        raw=json.dumps(report).encode();worker.write_bytes(raw)
        row['sha256']=hashlib.sha256(raw).hexdigest();row['median_step_seconds']=5.
    summary['decisions']=decisions(summary['records']);path.write_text(json.dumps(summary))
    result=audit(tmp_path)
    assert result['native_simple_vs_native_auto']['paired_throughput_ratios']==[1.2,1.2]
    assert 'native_simple_k36' in result['candidates_for_quality']
    assert 'triton_head_simple_k32' not in result['candidates_for_quality']
