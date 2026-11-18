"""Training entry points should share the same numerical checks."""
import importlib
import sys
from pathlib import Path

SCRIPTS = str(Path(__file__).resolve().parents[1] / 'scripts')
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_training_entries_use_shared_gradient_check():
    shared = importlib.import_module('qwen3_triton.validation.training_reference')
    reference = importlib.import_module('export_training_reference')
    runtime = importlib.import_module('train')
    assert reference.compare_first_gradients is shared.compare_first_gradients
    assert runtime.compare_first_gradients is shared.compare_first_gradients
    assert reference.sha is shared.sha


def test_shared_gradient_check_accepts_match_and_rejects_changed_gradient(tmp_path, monkeypatch):
    import json
    import torch

    shared = importlib.import_module('qwen3_triton.validation.training_reference')
    params = {f'weight_{i}': torch.nn.Parameter(torch.ones(1)) for i in range(399)}
    for param in params.values():
        param.grad = torch.ones(1)
    ids = torch.tensor([[1, 2]])
    logits = torch.tensor([[1.0, 2.0]])
    loss = torch.tensor(0.5)
    raw = {'ids': ids, 'gradients': {name: p.grad.clone() for name, p in params.items()},
           'logits': logits, 'local_loss': loss}
    result = {'input_manifest_sha256': 'input',
              'math_policy': {'allow_bf16_reduced_precision_reduction': False},
              'variant': 'hf', 'layers': 36,
              'ranks': [{'reference_archive': {'sha256': 'archive'}}]}
    (tmp_path / 'result.json').write_text(json.dumps(result))
    monkeypatch.setattr(shared, 'sha', lambda path: 'archive')
    monkeypatch.setattr(shared, 'load_archive', lambda path: {'1': raw})
    check = lambda: shared.compare_first_gradients(params, ids, logits, loss, tmp_path, 0, 'input')
    assert check()['numerical_gate_passed']
    params['weight_0'].grad.add_(1)
    assert not check()['numerical_gate_passed']


def test_reference_evaluation_keeps_peer_memory_scope(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    reference = importlib.import_module('export_training_reference')
    events = []

    @contextmanager
    def memory():
        events.append('enter')
        yield
        events.append('exit')

    def core(model, dataset, is_triton):
        assert events == ['enter']
        return {'loss': 1.0}

    monkeypatch.setattr(reference, '_evaluate_core', core)
    model = SimpleNamespace(_peer_gradient_communication=SimpleNamespace(validation_memory=memory))
    assert reference.evaluate(model, None, True) == {'loss': 1.0}
    assert events == ['enter', 'exit']
