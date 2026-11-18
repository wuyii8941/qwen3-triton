"""Final commands consume stable configuration, never result artifacts."""
import ast
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
