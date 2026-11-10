import pytest
import scripts.screen_fsdp_1p7b_accumulation as module


def test_commands_preserve_effective_batch_and_existing_fp32_policy(tmp_path):
    for name,spec in module.SPECS.items():
        batch,accum,sync=spec;cmd=module.command(spec,tmp_path/name)
        assert batch*accum==2
        assert cmd[cmd.index('--batch')+1]==str(batch)
        assert cmd[cmd.index('--accumulation')+1]==str(accum)
        assert cmd[cmd.index('--gradient-sync')+1]==sync
        assert cmd[cmd.index('--keep-full-layers')+1]=='28'
        assert cmd[cmd.index('--memory-budget-gib')+1]=='24'
        assert '--exchange-reduce' not in cmd and '--reference-check' in cmd


def test_specialized_audit_requires_both_rank_hashes(monkeypatch):
    monkeypatch.setattr(module,'audit_result',lambda *a,**k:('base',))
    record=dict(global_parameters=1720574976,world_size=2,effective_global_batch=4,seq=512,
        memory_budget_gib=24,steps=12,warmup=3,manifest={'input_sha256_by_rank':['a'*64,'b'*64]},
        communication_environment={'NCCL_PROTO':None},exchange_callback_counts_by_rank=[{},{}])
    assert module.audit_model(record,module.SPECS['large_batch'])==('base',)
    record['manifest']['input_sha256_by_rank'].pop()
    with pytest.raises(ValueError):module.audit_model(record,module.SPECS['large_batch'])
