"""The tracked tree contains one unversioned runnable implementation."""
import ast
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = {
    "prepare_corpus.py", "configure_inputs.py", "export_training_reference.py",
    "train.py", "generate.py", "verify_inference.py",
}


def tracked():
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def test_only_final_public_scripts_remain():
    scripts = {Path(path).name for path in tracked() if path.startswith("scripts/") and path.endswith(".py")}
    assert scripts == PUBLIC
    for name in PUBLIC:
        assert not (ROOT / "scripts" / name).is_symlink()


def test_historical_artifacts_are_not_tracked():
    paths = tracked()
    assert not any(path.startswith("results/") and path.endswith(".json") for path in paths)
    assert not any(re.search(r"(?:^|/)[^/]*_v\d+(?:\.[^/]*)?$", path) for path in paths)
    assert not any(
        path.startswith(prefix)
        for path in paths
        for prefix in ("benchmarks/", "examples/", "model/", "ops/")
    )


def test_package_does_not_import_scripts():
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
