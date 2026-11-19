"""Fail closed if fixed model/corpus bytes drift from a verified manifest."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def validate(path, protocol_path, cfg):
    raw = Path(path).read_bytes()
    report = json.loads(raw)
    if report.get('verified') is not True or report.get('repo') != cfg.get('model_repo') or report.get('revision') != cfg.get('model_revision'):
        raise ValueError('Wrong or unverified model manifest')
    if report.get('protocol_sha256') != digest(protocol_path):
        raise ValueError('Protocol differs from frozen inputs')
    model = Path(cfg['model']).resolve()
    corpus = Path(cfg['corpus']).resolve()
    if str(model) != report.get('model_dir') or str(corpus) != report.get('corpus_dir'):
        raise ValueError('Frozen paths differ from runtime paths')
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    if not shards or shards != set(report['shards']):
        raise ValueError('Unexpected shard set')
    entries = [(model, name, entry['sha256']) for name, entry in report['shards'].items()]
    entries += [(model, name, expected) for name, expected in report['model_files_sha256'].items()]
    entries += [(corpus, name, expected) for name, expected in report['tokens_sha256'].items()]
    if set(report['tokens_sha256']) != {'train.bin', 'validation.bin', 'test.bin'}:
        raise ValueError('Missing corpus split')
    if set(report['model_files_sha256']) != {'config.json', 'tokenizer.json', 'model.safetensors.index.json'}:
        raise ValueError('Missing model configuration or tokenizer')
    for root, name, expected in entries:
        if Path(name).name != name or digest(root / name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    return hashlib.sha256(raw).hexdigest()
