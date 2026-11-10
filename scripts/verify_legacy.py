"""Verify archived source bytes without importing historical training modules."""
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Historical source changed: {name}')
    return len(manifest['files'])


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'experiments/legacy'
    print(f'Historical source checks passed: {verify(root)} files')
