import json
from pathlib import Path
from training.convergence_rules import Plateau


def test_preregistered_protocol_separates_plateau_and_safety():
    cfg=json.loads(Path('training/fsdp_convergence_protocol_v1.json').read_text())
    assert cfg['validation']['scope'].startswith('all complete')
    assert cfg['validation']['exclude_padding']
    assert not cfg['safety_cutoff_means_converged']
    rule=Plateau(**cfg['plateau'])
    stopped=[]
    for _ in range(cfg['plateau']['min_checks']):stopped.append(rule.update(2.0)[1])
    assert not any(stopped[:-1])
    assert stopped[-1]
    assert rule.update(1.0)==(True,False)
