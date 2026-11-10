"""Strict history/coverage audit, not an independent rerun of training."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from training.convergence_rules import Plateau


def audit(result, protocol, validation_blocks, test_blocks):
    if result.get('protocol') != protocol or result.get('synthetic_preflight'):
        raise ValueError('Frozen real-data protocol required')
    if (result.get('complete') is not True or result.get('converged') is not True
            or result.get('stop_reason') != 'validation_plateau'
            or result.get('best_parameters_restored') is not True):
        raise ValueError('Completed plateau with best-state restoration required')
    training = result['training']; validation = result['validation']
    updates = len(training); every = protocol['validation']['every_updates']
    if (not updates or updates > protocol['safety']['max_updates']
            or updates % every or validation_blocks < 1 or test_blocks < 1):
        raise ValueError('Invalid stop update or corpus coverage')
    for step, row in enumerate(training, 1):
        if row['step'] != step or not all(math.isfinite(row[k]) for k in
                ('loss','seconds','max_rank_peak_allocated_gib','max_rank_peak_reserved_gib')):
            raise ValueError('Nonfinite or discontinuous training history')
        if not (row['seconds'] > 0 and 0 <= row['max_rank_peak_allocated_gib']
                <= row['max_rank_peak_reserved_gib']):
            raise ValueError('Invalid training timing or memory')
    if [r['step'] for r in validation] != list(range(0, updates+1, every)):
        raise ValueError('Missing/extra validation checks')
    seq = protocol['sequence_length']; plateau = Plateau(**protocol['plateau'])
    elapsed = -1
    for i, row in enumerate(validation):
        if (row['blocks'] != validation_blocks or row['tokens'] != validation_blocks*(seq-1)
                or not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds'] < elapsed):
            raise ValueError('Invalid validation coverage or time order')
        elapsed = row['elapsed_seconds']
        _, done = plateau.update(row['loss'])
        if done != (i == len(validation)-1):
            raise ValueError('Must stop at first plateau, not an earlier/later check')
    best = min(validation, key=lambda r: r['loss'])
    if result['best_step'] != best['step'] or result['best_validation'] != best['loss']:
        raise ValueError('Actual minimum validation checkpoint required')
    if result['plateau_reference_loss'] != plateau.best:
        raise ValueError('Plateau reference mismatch')
    test = result['test']
    if (test['blocks'] != test_blocks or test['tokens'] != test_blocks*(seq-1)
            or not math.isfinite(test['loss'])):
        raise ValueError('Incomplete held-out test')
    if not math.isfinite(result['elapsed_seconds']) or result['elapsed_seconds'] < elapsed:
        raise ValueError('Invalid total job duration')
    return dict(backend=result['backend'], history_consistent=True, updates=updates,
        validation_checks=len(validation), best_step=best['step'], best_validation=best['loss'],
        test=test, scope='Recorded history/coverage consistency; not independent training replay or exported-weight verification')


def main():
    p = argparse.ArgumentParser(); p.add_argument('results', nargs='+', type=Path)
    p.add_argument('--protocol',type=Path,default=Path('training/fsdp_convergence_protocol_v1.json'))
    p.add_argument('--output',type=Path,required=True); args = p.parse_args()
    protocol=json.loads(args.protocol.read_text()); corpus=Path(protocol['corpus'])
    def blocks(name):
        size=(corpus/name).stat().st_size
        if size % 4: raise ValueError('Invalid uint32 corpus file')
        return size//4//protocol['sequence_length']
    rows=[]
    for path in args.results:
        raw=path.read_bytes()
        rows.append(dict(source=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            **audit(json.loads(raw),protocol,blocks('validation.bin'),blocks('test.bin'))))
    with args.output.open('x') as stream: json.dump(rows,stream,indent=2);stream.write('\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
