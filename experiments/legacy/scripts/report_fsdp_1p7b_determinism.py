"""Fresh-process audit of raw deterministic diagnostic records."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_fsdp_exchange_confirmation import loss_comparison
from scripts.summarize_fsdp_memory import audit_result


def summarize(root):
    raw_summary=(root/'result.json').read_bytes();summary=json.loads(raw_summary)
    names={'native_a':(0,0,'full'),'native_b':(0,0,'full'),
        'keep28':(28,0,'full'),'checkpoint28':(28,1,'full')}
    if summary.get('complete') is not True or set(summary['provenance'])!=set(names):
        raise ValueError('All four complete diagnostic runs required')
    signatures=set();runs={}
    for name,policy in names.items():
        provenance=summary['provenance'][name]
        path=root/name/'result.json';raw=path.read_bytes();result=json.loads(raw)
        log=(root/(name+'.log')).read_bytes()
        if (Path(provenance['path']).resolve()!=path.resolve() or
                hashlib.sha256(raw).hexdigest()!=provenance['sha256'] or
                hashlib.sha256(log).hexdigest()!=provenance['log_sha256']):
            raise ValueError('Changed raw diagnostic evidence')
        signature=audit_result(result,policy)
        manifest=result['manifest'];diagnostic=result['deterministic_diagnostic']
        if (result['global_parameters']!=1720574976 or result['world_size']!=2 or
                result['seq']!=512 or result['batch']!=1 or result['warmup']!=1 or
                result['steps']!=3 or diagnostic.get('enabled') is not True or
                diagnostic['cublas_workspace']!=':4096:8' or
                diagnostic['source_hashes']['deterministic_fsdp_comparison.py']!=summary['wrapper_sha256'] or
                diagnostic['source_hashes']['compare_fsdp_training.py']!=manifest['source_sha256']):
            raise ValueError('Wrong model or deterministic contract')
        if (manifest['implementation_sources']!=summary['implementation_sources'] or
                len(manifest['input_sha256_by_rank'])!=2):raise ValueError('Sources/inputs mismatch')
        signatures.add(signature+(tuple(manifest['input_sha256_by_rank']),))
        runs[name]=result
        if summary['losses'][name]!=[r['loss'] for r in result['records']]:
            raise ValueError('Summary loss values differ from raw records')
    if len(signatures)!=1:raise ValueError('Matched diagnostic workload required')
    checks={n:loss_comparison(runs[n],runs['native_a']) for n in names if n!='native_a'}
    exact={n:all(a['loss']==b['loss'] for a,b in zip(runs[n]['records'],runs['native_a']['records']))
        for n in names if n!='native_a'}
    if checks!=summary['checks'] or exact['native_b']!=summary['native_repeat_losses_exact']:
        raise ValueError('Diagnostic conclusions differ from raw evidence')
    return dict(complete=True,raw_summary_path=str((root/'result.json').resolve()),
        raw_summary_sha256=hashlib.sha256(raw_summary).hexdigest(),
        checked_runs=list(names),loss_exact_vs_native=exact,loss_checks=checks,
        frozen_content_sha256=summary['frozen_content_sha256'],
        implementation_sha256=summary['implementation_sources']['sha256'],
        defaults_changed=False,convergence_proven=False,
        scope='Independent record audit: four real1.7B updates, mean losses only; not full gradients/moments, 15-step replay or performance')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=summarize(args.root)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
