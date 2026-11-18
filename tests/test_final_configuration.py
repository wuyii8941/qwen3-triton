"""Final commands consume stable configuration, never result artifacts."""
import ast
import re
import tomllib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_training_configuration_is_self_contained():
    protocol = json.loads((ROOT / "configs/fsdp_8b.json").read_text())
    assert protocol["model_repo"] == "Qwen/Qwen3-8B"
    assert "qwen_operator_lab" not in json.dumps(protocol)
    plan = ROOT / "configs/gemm_plan.json"
    assert plan.is_file()
    source = (ROOT / "scripts/train.py").read_text()
    assert "configs/gemm_plan.json" in source
    assert "results/" not in source


def test_retained_python_has_no_result_json_dependency():
    public = (
        "prepare_corpus.py", "configure_inputs.py", "export_training_reference.py",
        "train.py", "generate.py", "verify_inference.py",
    )
    for name in public:
        path = ROOT / "scripts" / name
        ast.parse(path.read_text())
        assert "results/" not in path.read_text(), path


def test_pyproject_matches_pinned_runtime_requirements():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = (ROOT / "requirements-qwen3-triton.txt").read_text().splitlines()

    def package_name(spec: str) -> str:
        return re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].lower().replace("_", "-")

    declared = {package_name(spec): spec for spec in project["dependencies"]}
    pinned = {
        package_name(spec): spec
        for line in requirements
        if (spec := line.strip()) and not spec.startswith("#")
    }
    assert set(declared) == set(pinned)
