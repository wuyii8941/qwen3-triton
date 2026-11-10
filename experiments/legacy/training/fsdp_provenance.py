"""Small source snapshots, excluding data, weights, caches and artifacts."""
import hashlib
import json
from pathlib import Path


def source_snapshot(root):
    root = Path(root).resolve()
    files = {root/'benchmarks/compare_fsdp_training.py'}
    for directory in ('training', 'ops', 'model'):
        files.update((root/directory).rglob('*.py'))
    hashes = {}
    for path in sorted(files):
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('Source symlink escapes project root')
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), files=hashes)
