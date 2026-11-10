"""Pinned download into local models directory; weights are never Git tracked."""
import json
import os
import subprocess
import hashlib
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    proxy = 'http://127.0.0.1:27891'
    for key in ('http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY'):
        os.environ[key] = proxy
    def curl(*args):
        return subprocess.check_output(['curl','-x',proxy,'-fLsS','--connect-timeout','20',
            '--retry','3','--retry-delay','2',*args])
    metadata = json.loads(curl('--max-time','90','https://huggingface.co/api/models/Qwen/Qwen3-8B?blobs=true'))
    revision = metadata['sha']
    target = Path('/data1/tzh/models/Qwen/Qwen3-8B')
    target.mkdir(parents=True,exist_ok=True)
    (target/'download_revision.json').write_text(json.dumps(dict(repo='Qwen/Qwen3-8B',revision=revision)))
    print(f'Downloading pinned Qwen/Qwen3-8B revision {revision}',flush=True)
    def segmented(item,path):
        """Bounded 16 total streams; validate each range and final LFS hash."""
        chunk_size = 64*1024*1024
        size = item['lfs']['size']
        chunks = path.with_suffix(path.suffix+'.chunks')
        chunks.mkdir(exist_ok=True)
        count = (size+chunk_size-1)//chunk_size
        old = path.with_suffix(path.suffix+'.part')
        # Retain existing sequential download bytes as chunk prefixes.
        if old.exists():
            with old.open('rb') as source:
                for index in range(count):
                    data = source.read(chunk_size)
                    if not data: break
                    part = chunks/str(index)
                    if not part.exists() or part.stat().st_size < len(data):
                        part.write_bytes(data)
        def fetch(index):
            start, end = index*chunk_size, min((index+1)*chunk_size,size)-1
            part = chunks/str(index)
            for attempt in range(60):
                have = part.stat().st_size if part.exists() else 0
                if have == end-start+1: return
                if have > end-start+1: raise RuntimeError('Oversized range fragment')
                tail = chunks/f'{index}.transfer'
                try:
                    status = curl('--max-time','1800','--range',f'{start+have}-{end}',
                        '-w','%{http_code}','-o',str(tail),
                        f'https://huggingface.co/Qwen/Qwen3-8B/resolve/{revision}/{path.name}?chunk={index}&t={int(time.time())}')
                except subprocess.CalledProcessError as error:
                    status = error.output
                if status == b'206' and tail.exists() and tail.stat().st_size <= end-start+1-have:
                    with part.open('ab') as output,tail.open('rb') as source:
                        shutil.copyfileobj(source,output)
                if tail.exists(): tail.unlink()
                if attempt == 59: raise RuntimeError(f'Range retries exhausted: {path.name}/{index}')
                time.sleep(1)
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(fetch,range(count)))
        assembled = path.with_suffix(path.suffix+'.assembled')
        with assembled.open('wb') as output:
            for index in range(count):
                with (chunks/str(index)).open('rb') as source: shutil.copyfileobj(source,output)
        digest = hashlib.sha256()
        with assembled.open('rb') as source:
            for block in iter(lambda:source.read(8*1024*1024),b''): digest.update(block)
        if digest.hexdigest() != item['lfs']['sha256']:
            raise RuntimeError(f'Assembled hash mismatch: {path.name}')
        assembled.rename(path)
        for index in range(count): (chunks/str(index)).unlink()
        chunks.rmdir()
        if old.exists(): old.unlink()
    def download(item):
        name = item['rfilename']
        if '/' in name or not name.endswith(('.json','.safetensors','.txt','.jinja','LICENSE')):
            return
        path = target/name
        if not path.exists() and 'lfs' in item and name.endswith('.safetensors'):
            print(f'Segmented download {name}',flush=True)
            segmented(item,path)
        if not path.exists():
            partial = path.with_suffix(path.suffix+'.part')
            print(f'Downloading {name}',flush=True)
            for attempt in range(30):
                try:
                    curl('--max-time','7200','--speed-limit','1024','--speed-time','120',
                        '-C','-','-o',str(partial),
                        f'https://huggingface.co/Qwen/Qwen3-8B/resolve/{revision}/{name}?download=true')
                    break
                except subprocess.CalledProcessError:
                    if attempt == 29: raise
                    print(f'Resume {name}, attempt {attempt+2}',flush=True)
                    time.sleep(2)
            partial.rename(path)
        if 'lfs' in item:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024),b''): digest.update(block)
            if path.stat().st_size != item['lfs']['size'] or digest.hexdigest() != item['lfs']['sha256']:
                raise RuntimeError(f'File integrity check failed: {name}')
        print(f'Verified {name}',flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(download,metadata['siblings']))
    (target/'download_complete.json').write_text(json.dumps(dict(repo='Qwen/Qwen3-8B',revision=revision,verified=True)))
    print('Download complete',flush=True)


if __name__ == '__main__': main()
