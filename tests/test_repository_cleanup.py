import importlib.util
import subprocess
import sys


def test_training_helpers_are_in_the_package():
    assert importlib.util.find_spec("qwen3_triton.training.data") is not None


def test_reference_exporter_exposes_only_the_native_reference():
    result = subprocess.run(
        [sys.executable, "scripts/export_training_reference.py", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--variant {hf}" in result.stdout
    assert "--save-reference" in result.stdout
    assert "ablate_" not in result.stdout
    assert "--vocab-ce" not in result.stdout
