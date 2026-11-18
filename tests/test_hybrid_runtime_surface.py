from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_cli_exposes_only_explicit_collective_modes():
    source = (ROOT / "scripts" / "train.py").read_text()
    assert "--collective-mode" in source
    assert "choices=['full','hybrid']" in source
    assert "TrainingRuntime(args.collective_mode" in source


def test_runtime_constructs_a_separate_nccl_group_for_hybrid_mode():
    source = (ROOT / "qwen3_triton" / "runtime" / "manager.py").read_text()
    assert "HybridCollectives" in source
    assert "backend='nccl'" in source
    assert "kind not in ('full','hybrid')" in source
    assert "scatter=370*2**20" in source
    assert "gather=4096" in source
