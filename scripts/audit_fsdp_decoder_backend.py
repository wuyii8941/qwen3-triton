"""Independently re-read fixed-K32 local reduction backend evidence."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.screen_fsdp_decoder_backend import ORDER, audit_gate, audit_run, helpers, selection
from scripts.audit_fsdp_retention_pressure import audit_resources
from training.fsdp_provenance import source_snapshot


def audit(root, *, allow_partial=False):
    complete = (root / 'result.json').exists()
    if not complete and not allow_partial:
        raise ValueError('Backend screen incomplete')
    result = json.loads((root / 'result.json').read_text()) if complete else None
    if complete and (result.get('complete') is not True or result['full_quality_completed'] is not False
            or result['defaults_changed'] is not False or result['helper_sources'] != helpers()
            or result['implementation_sources'] != source_snapshot(Path.cwd())
            or json.loads((root / 'status.json').read_text())['phase'] != 'finished'):
        raise ValueError('Complete matched-source backend screen required')
    gate_raw = (root / 'gate.json').read_bytes()
    gate = json.loads(gate_raw)
    gate_log = (root / 'gate.log').read_bytes()
    rows = audit_gate(gate_log.decode())
    if (gate['passed'] is not True or gate['records'] != rows or gate['source_sha256'] != helpers()
            or gate['log_sha256'] != hashlib.sha256(gate_log).hexdigest()
            or any(row['dtype'] != 'torch.bfloat16' or row['accumulation'] != 1 for row in rows)):
        raise ValueError('Numerical gate/source/log mismatch')
    if complete and result['gate'] != gate:
        raise ValueError('Unbound numerical gate')
    audit_resources(json.loads((root / 'gate_resources.json').read_text()))
    recorded = result['records'] if complete else json.loads((root / 'progress.json').read_text())
    order = [row['case'] for row in recorded]
    if not order or order != list(ORDER[:len(order)]) or (complete and len(order) != 4):
        raise ValueError('Exact ABBA order and coverage required')
    signatures, evidence = set(), []
    for row in recorded:
        name = row['case']
        signature, actual = audit_run(root / name, name)
        signatures.add(signature)
        actual.update(source=str(root / name), log_sha256=hashlib.sha256((root / f'{name}.log').read_bytes()).hexdigest())
        if row != actual:
            raise ValueError('Stored summary differs from raw backend evidence')
        worker = json.loads((root / name / 'result.json').read_text())
        if worker['manifest']['implementation_sources'] != source_snapshot(Path.cwd()):
            raise ValueError('Worker source mismatch')
        resources_raw = (root / f'{name}_resources.json').read_bytes()
        audit_resources(json.loads(resources_raw))
        evidence.append(dict(row, resource_sha256=hashlib.sha256(resources_raw).hexdigest()))
    if len(signatures) != 1:
        raise ValueError('Unmatched workload/input/source signatures')
    decision = selection(recorded) if complete else None
    if complete and result['selection'] != decision:
        raise ValueError('Selection does not follow raw measurements')
    return dict(complete=complete, partial_diagnostic_only=not complete,
        gate_sha256=hashlib.sha256(gate_raw).hexdigest(), raw_jobs_audited=len(recorded),
        records=evidence, selection=decision, full_quality_completed=False, defaults_changed=False,
        scope='Fixed-K32 ABBA short performance screen plus tied/untied numerical gate; not convergence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report = audit(args.root, allow_partial=args.allow_partial)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
