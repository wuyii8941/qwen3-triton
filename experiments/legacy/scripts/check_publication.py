"""Check publishable paths, Markdown links, file sizes and obvious credentials."""
import argparse
from pathlib import Path
import re
import subprocess


SECRET = re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----')
WEIGHT_SUFFIXES = {'.pt', '.pth', '.safetensors', '.bin', '.ckpt', '.npy', '.npz', '.pkl', '.pickle', '.pem', '.key'}


def check(root, *, staged=False):
    command = (['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z'] if staged else
               ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'])
    names = sorted(set(subprocess.check_output(command, cwd=root).decode().split('\0')) - {''})
    errors, count = [], 0
    for name in names:
        path = root / name
        if not path.is_file():
            continue  # Tracked deletions are not publication candidates.
        count += 1
        if (name.startswith(('submission/', 'artifacts/'))
                or any(part.startswith(('.venv', '.runtime', '__pycache__')) for part in path.relative_to(root).parts)
                or path.suffix in WEIGHT_SUFFIXES or (path.name.startswith('.env') and path.name != '.env.example')):
            errors.append(f'Forbidden publication path: {name}')
        if path.stat().st_size > 5 * 2**20:
            errors.append(f'File exceeds 5 MiB review limit: {name}')
            continue
        text = path.read_bytes().decode('utf-8', errors='replace')
        if SECRET.search(text):
            errors.append(f'Possible credential: {name}')
        if path.suffix == '.md':
            for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
                target = target.strip('<>').split('#', 1)[0]
                if not target or '://' in target or target.startswith('mailto:'):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.is_relative_to(root) or not resolved.exists():
                    errors.append(f'Broken or nonportable Markdown link: {name}: {target}')
    if errors:
        raise ValueError('\n'.join(errors))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(f'Publication checks passed: {check(root, staged=args.staged)} files')


if __name__ == '__main__':
    main()
