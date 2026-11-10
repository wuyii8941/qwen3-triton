"""Audit actual terminal jobs, frozen protocol, corpus and checkpoint metadata.

Run only on trusted local checkpoints. Metadata loading does not prove tensor
values are finite or bitwise equivalent; real terminal evaluation is separate
evidence. This verifier never changes a training job or stopping rule.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.report_convergence_comparison import audit, events
from training.convergence_rules import Plateau


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root):
    protocol_path = root / 'frozen_protocol.json'
    cfg = json.loads(protocol_path.read_text())
    protocol_hash = sha(protocol_path)
    corpus = Path(cfg['corpus'])
    manifest_path = corpus / 'tokens_manifest.json'
    assert sha(manifest_path) == cfg['data_manifest_sha256'], 'Dataset manifest changed'
    manifest = json.loads(manifest_path.read_text())
    data_checks = {}
    for name, entry in manifest['splits'].items():
        path = Path(entry['token_file'])
        assert sha(path) == entry['token_sha256'], f'{name} token contents changed'
        assert path.stat().st_size == entry['tokens'] * 4
        data_checks[name] = dict(tokens=entry['tokens'], sha256=entry['token_sha256'],
                                 complete_blocks=entry['tokens'] // cfg['seq'],
                                 dropped_tail_tokens=entry['tokens'] % cfg['seq'])
    report = audit(root)
    assert report['finished'] and report['converged_both'], 'Actual convergence missing'
    checkpoint_checks = {}
    for name in ('native', 'triton'):
        directory = root / name
        rows = events(directory / 'events.jsonl')
        summary = report['jobs'][name]['summary']
        assert summary['protocol_hash'] == protocol_hash
        assert rows[-1]['event'] == 'terminal'
        assert rows[-1]['step'] == summary['step']
        assert summary['reason'] == 'validation_plateau'
        plateau = Plateau(**cfg['plateau'])
        validation = [r for r in rows if r['event'] == 'validation']
        assert [r['step'] for r in validation] == list(range(cfg['eval_every'], summary['step'] + 1, cfg['eval_every']))
        for index, row in enumerate(validation):
            _, stopped = plateau.update(row['loss'])
            assert asdict(plateau) == row['plateau'], 'Plateau history mismatch'
            assert stopped == (index == len(validation) - 1), 'Job stopped too early/late'
        assert asdict(plateau) == summary['plateau']
        for filename, step in [('last.pt', summary['step']), ('best.pt', summary['best_step'])]:
            path = directory / filename
            # No giant CPU allocations or GPU work needed to inspect metadata.
            state = torch.load(path, map_location='meta', weights_only=False)
            assert state['step'] == step and state['protocol_hash'] == protocol_hash
            assert state['optimizer_name'] == name
            assert state['model'] and state['optimizer']['state']
            checkpoint_checks[f'{name}/{filename}'] = dict(step=step, bytes=path.stat().st_size,
                model_tensors=len(state['model']), optimizer_states=len(state['optimizer']['state']))
            del state
    report.update(protocol_sha256=protocol_hash, dataset_integrity=data_checks,
                  checkpoint_metadata=checkpoint_checks,
                  verification='Terminal events, plateau replay, protocol/corpus hashes, checkpoint metadata verified.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    with args.output.open('x') as destination:
        json.dump(result, destination, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k.endswith('speedup') or k == 'verification'}, indent=2))
