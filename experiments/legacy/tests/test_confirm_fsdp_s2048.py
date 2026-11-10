import json
import pytest
from scripts.confirm_fsdp_s2048 import choose_policies,decisions,check_runtime,benchmark_command,verify_numerical_gate


def native(name,k,every,seconds,status='passed'):
    return dict(case=name,status=status,median_step_seconds=seconds,
        policy=dict(keep_full_layers=k,checkpoint_every=every,checkpoint_mode='full'))


def test_select_strongest_non_ce_and_deduplicate_baseline():
    n=dict(records=[native('full_k8',8,2,7.4),native('all_k24',24,1,7.),native('oom',30,1,0,'oom')])
    c=dict(records=[dict(case='ce12',status='passed',policy=[12,2,'full',128],median_step_seconds=7.2)])
    selected=choose_policies(n,c)
    assert selected['no_ce']=='noce_e1_k24' and selected['candidate']=='noce_e1_k24'
    assert len(selected['policies'])==3
    selected=choose_policies(dict(records=n['records'][:1]),c)
    assert selected['no_ce']==selected['baseline'] and len(selected['policies'])==2
    with pytest.raises(ValueError):choose_policies(dict(records=[]),c)


def test_ce_adoption_requires_both_orders_not_cherry_picked_round():
    selection=dict(baseline='base',no_ce='native',ce='ce')
    rows=[dict(case=f'r{r}_{name}',median_step_seconds=value) for r,values in enumerate(
        (dict(base=8,native=7,ce=6.8),dict(base=8,native=7,ce=7.1))) for name,value in values.items()]
    result=decisions(dict(records=rows),selection)
    assert not result['ce_over_two_percent_in_both_orders']
    assert result['candidate_for_quality_validation']=='native'
    assert result['no_ce_over_two_percent_in_both_orders'] and not result['defaults_changed']


def test_runtime_must_match_frozen_versions(monkeypatch):
    import scripts.confirm_fsdp_s2048 as module
    monkeypatch.setattr(module.importlib.metadata,'version',lambda name:'v1')
    monkeypatch.setattr(module.platform,'python_version',lambda:'3.11')
    manifest=dict(packages={n:'v1' for n in ('torch','transformers','triton')},python='3.11')
    check_runtime(manifest)
    manifest['packages']['torch']='v2'
    with pytest.raises(ValueError):check_runtime(manifest)


def test_do_not_promote_ce_if_original_control_is_faster():
    selection=dict(baseline='base',no_ce='native',ce='ce')
    rows=[dict(case=f'r{r}_{name}',median_step_seconds=value) for r in (0,1)
          for name,value in dict(base=6,native=7,ce=6.8).items()]
    result=decisions(dict(records=rows),selection)
    assert not result['ce_over_two_percent_in_both_orders']
    assert result['candidate_for_quality_validation']=='base'


def test_headroom_and_confirmation_commands_preserve_policy(tmp_path):
    command=benchmark_command(dict(model='/data/model'),(36,1,'full',0),tmp_path/'case',6)
    assert command[command.index('--keep-full-layers')+1]=='36'
    assert command[command.index('--checkpoint-every')+1]=='1'
    assert command[command.index('--steps')+1]=='6'
    assert command[command.index('--ce-only-chunk-size')+1]=='0'
    assert '--reference-check' in command


def test_all_retained_gate_cannot_be_confused_with_mixed_retention(tmp_path):
    rows=[dict(passed=True,tied=t,backend='official',dtype='torch.bfloat16',world=4,head_dim=128,
        seq=2048,steps=3,deterministic=True,checkpoint_every=1,checkpoint_mode='full',
        keep_full_layers=2,loss_chunk_size=0,ce_only_chunk_size=0,swiglu=False,
        accumulation=1,gradient_sync='each') for t in (False,True)]
    path=tmp_path/'gate.log';path.write_text('\n'.join(json.dumps(r) for r in rows))
    verify_numerical_gate(path,1,0,keep=2)
    with pytest.raises(ValueError):verify_numerical_gate(path,1,0,keep=1)
