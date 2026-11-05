"""Convert existing public WikiText parquet cache; needs pyarrow, NOT torch.

Run with any existing pyarrow-capable Python, without changing lab dependencies.
This transforms data, not code, and refuses to overwrite previous artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result={'dataset':'Salesforce/wikitext','config':'wikitext-103-raw-v1',
            'revision':args.source.parent.name,'splits':{}}
    for split in ('validation','test'):
        path=args.source/f'{split}-00000-of-00001.parquet'
        result['splits'][split]={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                'text':'\n'.join(pq.read_table(path,columns=['text']).column('text').to_pylist())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f)
    print(json.dumps({s:len(v['text']) for s,v in result['splits'].items()}))


if __name__=='__main__':main()
