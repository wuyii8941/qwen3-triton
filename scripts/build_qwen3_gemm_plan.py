"""Build an explicit SM86 tuning plan from a clean completed operator screen.

Require improvements in both confirmation orders; never infer a training gain.
All rejected candidates remain in the input report. No GPU is initialized.
"""
from pathlib import Path
import argparse, hashlib, json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--minimum-reduction',type=float,default=.02)
    args=parser.parse_args()
    assert 0 < args.minimum_reduction < 1
    assert not args.output.resolve().is_relative_to(Path('/home/tzh'))
    report_path=args.screen/'measurements/result.json'
    monitor=json.loads((args.screen/'gpu.json').read_text())
    assert monitor['returncode']==0 and monitor['selected_device_clean']
    report=json.loads(report_path.read_text())
    assert report['complete'] and len(report['rows'])==16
    assert report['provenance']['capability']==[8,6]
    entries=[]
    for row in report['rows']:
        baseline=next(r for r in row['records'] if r['label']==row['baseline'])
        qualified=[]
        for candidate in row['records']:
            # Accumulator spelling is a negative control; its IR is equivalent.
            if candidate['family'] in ('v9','explicit_accum'):
                continue
            if not candidate.get('numerical_passed'):
                continue
            if any(f'confirmation_order{o}' not in candidate for o in (1,2)):
                continue
            reductions=[1-candidate[f'confirmation_order{o}']['median_ms']/baseline[f'confirmation_order{o}']['median_ms'] for o in (1,2)]
            if min(reductions) >= args.minimum_reduction:
                qualified.append((sum(reductions),candidate,reductions))
        if not qualified:
            continue
        _,candidate,reductions=max(qualified,key=lambda x:x[0])
        entry=dict(kind='gate_up' if row['direction']=='fused_forward' else 'gemm',
            M=row['M'],N=row['N'],K=row['K'],strides=row['strides'],
            family=candidate['family'],config=candidate['config'],splits=candidate['splits'],
            source_module=row['module'],source_direction=row['direction'],
            latency_reductions=reductions,numerical=candidate['numerical'],
            partial_workspace_bytes=candidate['partial_workspace_bytes'])
        entries.append(entry)
    keys=[(e['kind'],e['M'],e['N'],e['K'],json.dumps(e['strides'])) for e in entries]
    assert len(keys)==len(set(keys))
    result=dict(schema=1,hardware_capability=[8,6],compute_dtype='bfloat16',
        accumulation_dtype='float32',entries=entries,
        source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        resource_monitor_sha256=hashlib.sha256((args.screen/'gpu.json').read_bytes()).hexdigest(),
        minimum_reduction_in_each_order=args.minimum_reduction,
        complete_model_numerically_qualified=False,whole_step_performance_qualified=False,
        scope=__doc__)
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(plan=str(args.output),entries=len(entries),full_model_qualified=False)))


if __name__=='__main__':main()
