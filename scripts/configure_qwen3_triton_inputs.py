"""Verify pinned Qwen3-8B/corpus bytes and bind new local paths immutably.

Only model and corpus paths change from the frozen training protocol. Model
shards, configuration/tokenizer/index and all three token splits must match
the checked-in recipe. No network, downloads or existing-evidence edits.
"""
from pathlib import Path
import argparse,hashlib,json
from training.input_manifest import validate

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*2**20),b''):digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--corpus-dir',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,default=ROOT/'results/qwen3_triton_input_recipe_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();recipe=json.loads(args.recipe.read_text())
    model=args.model_dir.resolve();corpus=args.corpus_dir.resolve()
    if args.output.resolve().is_relative_to(Path('/home/tzh')):parser.error('Output under /home/tzh is forbidden')
    if args.output.exists():raise FileExistsError('Use a new output directory; existing evidence is immutable')
    official=recipe['model'];assert official['repo']=='Qwen/Qwen3-8B'
    assert official['revision']=='b968826d9c46dd6066d109eabc6255188de91218'
    for name,expected in official['model_files_sha256'].items():
        if Path(name).name!=name or sha(model/name)!=expected:raise ValueError('Wrong official8B model file: '+name)
    index=json.loads((model/'model.safetensors.index.json').read_text())
    if set(index['weight_map'].values())!=set(official['shards']):raise ValueError('Wrong model shard set')
    for name,entry in official['shards'].items():
        if Path(name).name!=name or (model/name).stat().st_size!=entry['size'] or sha(model/name)!=entry['sha256']:
            raise ValueError('Wrong official8B shard: '+name)
    tokens={}
    for split,entry in recipe['dataset']['splits'].items():
        name=split+'.bin';path=corpus/name
        if path.stat().st_size!=entry['tokens']*4 or sha(path)!=entry['token_sha256']:raise ValueError('Wrong frozen token bytes: '+split)
        tokens[name]=entry['token_sha256']
    protocol=json.loads((ROOT/'configs/fsdp_8b.json').read_text())
    assert protocol['model_repo']==official['repo'] and protocol['model_revision']==official['revision']
    protocol['model']=str(model);protocol['corpus']=str(corpus)
    protocol['status']='input_bytes_verified_numerical_and_performance_qualification_separate'
    args.output.mkdir(parents=True,exist_ok=False)
    protocol_path=args.output/'protocol.json'
    with protocol_path.open('x') as handle:handle.write(json.dumps(protocol,indent=2)+'\n')
    report=dict(repo=official['repo'],revision=official['revision'],protocol_sha256=sha(protocol_path),
        model_dir=str(model),shards=official['shards'],model_files_sha256=official['model_files_sha256'],
        corpus_dir=str(corpus),tokens_sha256=tokens,verified=True,
        token_provenance=dict(source_tokenizer_path=str(model/'tokenizer.json'),source_tokenizer_sha256=official['model_files_sha256']['tokenizer.json'],recipe_sha256=sha(args.recipe),provenance='All token bytes match the pinned corpus recipe. This path-binding operation does not regenerate the corpus.'),
        scope='Fixed official8B input fingerprint; not numerical correctness or performance',
        source_sha256=sha(Path(__file__)))
    path=args.output/'input_manifest.json'
    with path.open('x') as handle:handle.write(json.dumps(report,indent=2)+'\n')
    # Exercise the actual training validator, including a second independent
    # streaming pass, before reporting these files usable by the runner.
    identity=validate(path,protocol_path,protocol)
    print(json.dumps(dict(complete=True,verified=True,model='Qwen/Qwen3-8B',protocol=str(protocol_path),input_manifest=str(path),input_manifest_sha256=identity)),flush=True)

if __name__=='__main__':main()
