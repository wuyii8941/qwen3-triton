"""Recreate pinned WikiText tokens using the official Qwen3-8B tokenizer.

Supply cached parquet files, or verified exported text, and a new output
directory. No downloads, existing-input changes or home writes are performed.
Every completed split must reproduce its frozen text and token SHA256.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*2**20),b''):digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,default=ROOT/'results/qwen3_triton_input_recipe_v1.json')
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--parquet-source',type=Path)
    source.add_argument('--text-source',type=Path)
    parser.add_argument('--splits',nargs='+',choices=('train','validation','test'),default=['train','validation','test'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();recipe=json.loads(args.recipe.read_text())
    assert recipe['model']['repo']=='Qwen/Qwen3-8B'
    assert sha(args.model_dir/'tokenizer.json')==recipe['model']['model_files_sha256']['tokenizer.json']
    config=json.loads((args.model_dir/'config.json').read_text())
    assert sha(args.model_dir/'config.json')==recipe['model']['model_files_sha256']['config.json']
    assert config['hidden_size']==4096 and config['num_hidden_layers']==36
    if len(args.splits)!=len(set(args.splits)):parser.error('Each split can be requested once')
    if args.output.resolve().is_relative_to(Path('/home/tzh')):parser.error('Output under /home/tzh is forbidden')
    args.output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
    records={}
    for split in args.splits:
        expected=recipe['dataset']['splits'][split];text=args.output/(split+'.txt')
        if args.parquet_source:
            import pyarrow.parquet as pq
            for name,h in expected['parquet'].items():
                if sha(args.parquet_source/name)!=h:raise ValueError('Pinned parquet mismatch: '+name)
            with text.open('x',encoding='utf-8',newline='') as output:
                for name in sorted(expected['parquet']):
                    for batch in pq.ParquetFile(args.parquet_source/name).iter_batches(columns=['text'],batch_size=4096):
                        output.write('\n'.join(batch.column(0).to_pylist())+'\n')
        else:
            original=args.text_source/(split+'.txt')
            if sha(original)!=expected['text_sha256']:raise ValueError('Pinned text mismatch: '+split)
            shutil.copyfile(original,text)
        if sha(text)!=expected['text_sha256']:raise ValueError('Exported text mismatch: '+split)
        target=args.output/(split+'.bin');count=0
        with text.open(encoding='utf-8') as source,target.open('xb') as output:
            buffer=[];size=0
            def emit():
                nonlocal count
                ids=tokenizer(''.join(buffer),add_special_tokens=False,verbose=False)['input_ids']
                np.asarray(ids,dtype='<u4').tofile(output);count+=len(ids)
            for line in source:
                buffer.append(line);size+=len(line)
                if size>=65536:emit();buffer=[];size=0
            if buffer:emit()
        digest=sha(target)
        if count!=expected['tokens'] or target.stat().st_size!=count*4 or digest!=expected['token_sha256']:
            raise ValueError('Token replay differs from fixed corpus: '+split)
        records[split]=dict(text_sha256=sha(text),token_sha256=digest,tokens=count,exact_fixed_token_bytes=True)
        print(json.dumps(dict(split=split,**records[split])),flush=True)
    # A partial replay is explicitly labelled; it cannot replace all three
    # splits required by training.input_manifest.validate.
    result=dict(complete=True,all_three_splits_complete=set(records)=={'train','validation','test'},
        model='Qwen/Qwen3-8B',revision=recipe['model']['revision'],splits=records,
        tokenizer_sha256=sha(args.model_dir/'tokenizer.json'),recipe_sha256=sha(args.recipe),
        source_sha256=sha(Path(__file__)),scope=__doc__)
    with (args.output/'replay.json').open('x') as handle:handle.write(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
