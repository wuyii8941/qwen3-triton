"""Stream current reference/data hashes; never assert historic identity retroactively."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import stat


def identity(path):
    s=path.stat()
    if not stat.S_ISREG(s.st_mode):raise ValueError('Only regular reference files supported')
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)


def file_digest(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def snapshot(paths):
    paths={name:Path(path).resolve() for name,path in paths.items()}
    if not paths:raise ValueError('Nonempty explicit reference set required')
    before={name:identity(path) for name,path in paths.items()}
    rows={name:dict(path=str(path),bytes=before[name][2],mtime_ns=before[name][3],sha256=file_digest(path))
          for name,path in sorted(paths.items())}
    if before!={name:identity(path) for name,path in paths.items()}:
        raise ValueError('Reference changed during hashing; refuse a mixed snapshot')
    contents={name:dict(bytes=r['bytes'],sha256=r['sha256']) for name,r in rows.items()}
    encoded=json.dumps(contents,sort_keys=True,separators=(',',':')).encode()
    return dict(files=rows,content_sha256=hashlib.sha256(encoded).hexdigest())


def verify_frozen(report):
    if report.get('complete') is not True:raise ValueError('Complete manifest required')
    contents={name:dict(bytes=r['bytes'],sha256=r['sha256']) for name,r in report['files'].items()}
    encoded=json.dumps(contents,sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(encoded).hexdigest()!=report['content_sha256']:
        raise ValueError('Manifest content identity is internally inconsistent')
    current=snapshot({name:row['path'] for name,row in report['files'].items()})
    if current['content_sha256']!=report['content_sha256']:
        raise ValueError('Current reference bytes differ from frozen inputs')
    return True


def weight_path(root,name):
    root=Path(root).resolve();relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Weight index must contain relative in-root paths')
    resolved=(root/relative).resolve()
    if not resolved.is_relative_to(root) or resolved.suffix!='.safetensors':
        raise ValueError('Only in-root safetensors shards supported')
    return resolved


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,
        default=Path('training/fsdp_convergence_protocol_v1.json'))
    p.add_argument('--output',type=Path)
    p.add_argument('--verify',type=Path,help='Read-only full rehash against a captured manifest')
    a=p.parse_args()
    if a.verify:
        if a.output:p.error('Choose capture or verification, not both')
        verify_frozen(json.loads(a.verify.read_text()))
        print(json.dumps(dict(verified=True,manifest=str(a.verify))));return
    if a.output is None:p.error('--output required for capture')
    if a.output.exists():raise FileExistsError('Preserve frozen manifests')
    protocol=json.loads(a.protocol.read_text());model=Path(protocol['model']).resolve()
    corpus=Path(protocol['corpus']).resolve();index=model/'model.safetensors.index.json'
    mapping=json.loads(index.read_text())['weight_map']
    if not mapping:raise ValueError('Nonempty sharded checkpoint index required')
    paths={'protocol':a.protocol,'model/config.json':model/'config.json',
           'model/model.safetensors.index.json':index}
    for name in ('tokenizer.json','tokenizer_config.json'):
        paths['model/'+name]=model/name
    for name in sorted(set(mapping.values())):paths['model/'+name]=weight_path(model,name)
    for name in ('train.bin','validation.bin','test.bin'):paths['corpus/'+name]=corpus/name
    started=datetime.now(timezone.utc).isoformat();result=snapshot(paths)
    # The index parsed to discover shards must be the same version we hashed.
    if mapping!=json.loads(index.read_text())['weight_map']:
        raise ValueError('Weight index changed while enumerating references')
    if protocol!=json.loads(a.protocol.read_text()):
        raise ValueError('Protocol changed while enumerating references')
    from training.fsdp_provenance import source_snapshot
    report=dict(schema_version=1,complete=True,capture_started_utc=started,
        capture_finished_utc=datetime.now(timezone.utc).isoformat(),**result,
        implementation_sources=source_snapshot(Path.cwd()),python=platform.python_version(),
        packages={n:importlib.metadata.version(n) for n in ('torch','transformers','triton')},
        scope='Current checkpoint/protocol/corpus content identity only; not proof that historic jobs used identical files')
    with a.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(content_sha256=report['content_sha256'],files=len(paths),source=str(a.output))))


if __name__=='__main__':main()
