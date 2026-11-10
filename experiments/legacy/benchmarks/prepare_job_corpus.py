"""Export cached public parquet splits with an existing pyarrow Python.

Keeps training/validation/test strictly separate; no package installation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest={'dataset':'Salesforce/wikitext-103-raw-v1','revision':a.source.parent.name,'splits':{}}
    for split in ('train','validation','test'):
        files=sorted(a.source.glob(f'{split}-*.parquet'))
        if not files:raise FileNotFoundError(split)
        target=a.output/f'{split}.txt'
        with target.open('x') as out:
            for path in files:
                for batch in pq.ParquetFile(path).iter_batches(columns=['text'],batch_size=4096):
                    out.write('\n'.join(batch.column(0).to_pylist())+'\n')
        manifest['splits'][split]={'text':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'parquet':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
        print(split,target.stat().st_size,flush=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
