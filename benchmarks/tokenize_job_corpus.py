"""Stream fixed line-preserving text chunks to uint32 token shards."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);a=p.parse_args()
    tok=AutoTokenizer.from_pretrained('/data1/tzh/models/Qwen/Qwen3-1.7B',local_files_only=True)
    manifest=json.loads((a.corpus/'manifest.json').read_text())
    manifest['preprocessing']='UTF8 line-preserving chunks >=65536 chars, add_special_tokens=False; uint32 LE'
    for split in ('train','validation','test'):
        target=a.corpus/f'{split}.bin'
        if target.exists():raise FileExistsError(target)
        count=0
        with (a.corpus/f'{split}.txt').open() as source,target.open('xb') as out:
            buf=[];size=0
            def emit():
                nonlocal count
                ids=tok(''.join(buf),add_special_tokens=False,verbose=False)['input_ids']
                np.asarray(ids,dtype='<u4').tofile(out);count+=len(ids)
            for line in source:
                buf.append(line);size+=len(line)
                if size>=65536:emit();buf=[];size=0
            if buf:emit()
        manifest['splits'][split].update(token_file=str(target),tokens=count,
            token_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        print(split,count,flush=True)
    with (a.corpus/'tokens_manifest.json').open('x') as out:json.dump(manifest,out,indent=2)


if __name__=='__main__':main()
