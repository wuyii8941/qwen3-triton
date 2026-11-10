from pathlib import Path
from scripts.confirm_fsdp_s512_candidates import POLICIES, command


def test_preregistered_candidates_distinct_and_not_test_loss_selected():
    assert list(POLICIES)==['interval_k35','all_k32','all_k36']
    assert len(set(POLICIES.values()))==3
    assert POLICIES['interval_k35']==(35,2,'full')


def test_commands_hold_workload_and_reference_gate_fixed():
    for name, (keep,every,mode) in POLICIES.items():
        cmd=command(POLICIES[name],Path('/tmp')/name,'real-model')
        for option,value in {'--model':'real-model','--keep-full-layers':str(keep),
                '--checkpoint-every':str(every),'--checkpoint-mode':mode,'--seq':'512',
                '--batch':'1','--warmup':'3','--steps':'12','--memory-budget-gib':'42'}.items():
            assert cmd[cmd.index(option)+1]==value
        assert '--reference-check' in cmd
