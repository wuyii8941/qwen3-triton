import copy
import pytest
import scripts.confirm_fsdp_exchange as module


def test_commands_keep_strong_controls_and_scope(tmp_path):
    for name,spec in module.CASES.items():
        cmd=module.command(spec,tmp_path/name,'/data/model')
        assert cmd[cmd.index('--steps')+1]=='12'
        assert cmd[cmd.index('--checkpoint-every')+1]=='1'
        assert cmd[cmd.index('--keep-full-layers')+1]==str(spec[0])
        assert '--reference-check' in cmd
        assert ('--exchange-reduce' in cmd)==bool(spec[1])


def rows():
    return [dict(repeat=r,case=name,status='passed',median_step_seconds=seconds)
        for r in (0,1) for name,seconds in zip(module.CASES,(6.,5.,7.,5.5))]


def test_candidate_must_beat_stronger_auto_control():
    r=rows();out=module.decisions(r)
    assert out['triton_head_simple_k32']['ratios_vs_stronger_control']==[6/5.5]*2
    r[-1]['median_step_seconds']=6.5
    assert not module.decisions(r)['triton_head_simple_k32']['over_two_percent_both_orders']


def test_oom_is_not_speedup():
    r=rows();r[1]['status']='oom';r[1].pop('median_step_seconds')
    assert not module.decisions(r)['torch_decoder_auto_k32']['over_two_percent_both_orders']


def record():
    return dict(world_size=4,global_parameters=8190735360,seq=512,batch=1,steps=12,warmup=3,
        optimization=dict(exchange_reduce='torch',exchange_scope='decoder'),
        communication_environment={k:None for k in
            ('NCCL_PROTO','NCCL_ALGO','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE')},
        manifest=dict(input_sha256_by_rank=['a'*64]*4,exchange_sources={'adapter':'b'*64}),
        exchange_callback_counts_by_rank=[{'model.layers.0':15} for _ in range(4)])


@pytest.mark.parametrize('fault',['protocol','scope','missing_rank','inactive','wrong_model'])
def test_audit_rejects_mislabeled_candidate(monkeypatch,fault):
    monkeypatch.setattr(module,'audit_result',lambda *a,**k:('base',))
    good=record();assert module.audit_case(good,module.CASES['torch_decoder_auto_k32'])
    bad=copy.deepcopy(good)
    if fault=='protocol':bad['communication_environment']['NCCL_PROTO']='Simple'
    if fault=='scope':bad['optimization']['exchange_scope']='all'
    if fault=='missing_rank':bad['manifest']['input_sha256_by_rank'].pop()
    if fault=='inactive':bad['exchange_callback_counts_by_rank'][0]={}
    if fault=='wrong_model':bad['global_parameters']=123
    with pytest.raises(ValueError):module.audit_case(bad,module.CASES['torch_decoder_auto_k32'])
