"""Independently rebuild completed S2048 comparisons, gates and decisions."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.confirm_fsdp_s2048 import decisions,verify_numerical_gate
from scripts.summarize_fsdp_shape_confirmation import summarize


def audit(root,reference):
    root=Path(root)
    status=json.loads((root/'status.json').read_text())
    if status['phase']!='finished':raise ValueError('Supervisor is not terminal-successful')
    source=root/'result.json';raw=source.read_bytes();original=json.loads(raw)
    spec=json.loads((root/'spec.json').read_text());selection=json.loads((root/'selection.json').read_text())
    computed=summarize(root,spec)
    for key,value in computed.items():
        if original.get(key)!=value:raise ValueError(f'Supervisor report differs from raw evidence: {key}')
    decision=decisions(computed,selection)
    if decision!=original['decision'] or decision!=status['decision']:
        raise ValueError('Decision inconsistent with paired observations')
    if (reference.get('complete') is not True or
            reference['content_sha256']!=original['reference_content_sha256']):
        raise ValueError('Frozen reference identity mismatch')
    supervisor=Path('scripts/confirm_fsdp_s2048.py')
    if hashlib.sha256(supervisor.read_bytes()).hexdigest()!=original['supervisor_source_sha256']:
        raise ValueError('Supervisor source changed; archived source required for this audit')
    gates=[]
    for name,(keep,every,mode,chunk) in spec['policies'].items():
        if mode!='full':raise ValueError('Unexpected numerical gate policy')
        path=root/f'gate_{name}.log';verify_numerical_gate(path,every,chunk)
        gates.append(dict(policy=name,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    computed.update(decision=decision,reference_content_sha256=reference['content_sha256'],
        numerical_gates=gates,supervisor_result_source=str(source),
        supervisor_result_sha256=hashlib.sha256(raw).hexdigest(),
        supervisor_source_sha256=original['supervisor_source_sha256'],
        reference_verification_scope='Supervisor recorded successful full rehash before and after jobs; this audit does not rehash large inputs during subsequent timing')
    return computed


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.root,json.loads(a.reference.read_text()))
    with a.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report['decision'],indent=2))


if __name__=='__main__':main()
