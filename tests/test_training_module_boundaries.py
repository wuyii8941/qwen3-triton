"""Stable training helpers live in the package, never in launch scripts."""
import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stable_training_helpers_are_package_imports():
    model = importlib.import_module("qwen3_triton.training.model")
    reference = importlib.import_module("qwen3_triton.validation.training_reference")
    assert callable(model.shard_vocab_loss_model)
    for name in ("compare_first_gradients", "sha", "local"):
        assert callable(getattr(reference, name))


def test_package_never_imports_scripts():
    for path in (ROOT / "qwen3_triton").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name == "scripts" or name.startswith("scripts.") for name in names), path
