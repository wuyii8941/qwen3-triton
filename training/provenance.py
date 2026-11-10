"""Current-runtime source identity, independent of frozen legacy audits."""
import hashlib
import json
from pathlib import Path


def source_snapshot(root):
    root = Path(root).resolve()
    files = sorted(path for directory in ('training', 'ops', 'model', 'configs')
                   for path in (root / directory).rglob('*')
                   if path.is_file() and path.suffix in ('.py', '.json'))
    hashes = {}
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError('Source symlink escapes project root')
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), files=hashes)
