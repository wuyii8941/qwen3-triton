"""The final repository exposes six stable, CPU-safe commands."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (
    "prepare_corpus.py",
    "configure_inputs.py",
    "export_training_reference.py",
    "train.py",
    "generate.py",
    "verify_inference.py",
)


def test_public_commands_have_cpu_safe_help():
    for name in PUBLIC:
        path = ROOT / "scripts" / name
        assert path.is_file() and not path.is_symlink(), name
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, (name, result.stdout, result.stderr)
        assert "usage:" in result.stdout.lower(), name
