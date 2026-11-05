"""Verify evidence before claiming a complete-training contribution."""
import argparse
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def junit(path):
    root=ET.parse(path).getroot()
    suites=[root] if root.tag=='testsuite' else root.findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts['failures']==counts['errors']==counts['skipped']==0,counts
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    results={}
    for seq in (512,2048):
        path=Path(f'artifacts/full_training_repeated_adamw_rounded_s{seq}_v2.json')
        data=json.loads(path.read_text());assert data['adamw_math']=='rounded'
        assert len(data['runs'])==9
        rows={}
        inputs=set();devices=set()
        for run in data['runs']:
            assert run['returncode']==0
            r=json.loads(Path(run['artifact']).read_text());m=r['manifest']
            assert m['layers']==28 and m['seq']==seq and m['batch']==m['accumulation']==1
            assert m['checkpointing'] is False and m['use_cache'] is False
            assert len(r['samples'])==10 and r['scope'].startswith('steady')
            if run['variant']=='hf_adamw':assert m['adamw_math']=='rounded' and m['optimizer'].startswith('triton')
            else:assert m['optimizer'].startswith('native')
            if run['variant']=='compile':
                counts=r['compile_counters']
                assert counts['stats']['unique_graphs']==1
                assert not counts.get('graph_break',{}),'Compile graph breaks need separate accounting'
            inputs.add(m['input_sha256']);devices.add(m['visible_devices'])
            rows[(run['round'],run['variant'])]=r['medians']
        assert len(inputs)==len(devices)==1
        assert set(rows)=={(i,v) for i in range(3) for v in ('hf','compile','hf_adamw')}
        candidate=[rows[(i,'hf_adamw')]['step_ms'] for i in range(3)]
        ratios={v:[rows[(i,v)]['step_ms']/candidate[i] for i in range(3)] for v in ('hf','compile')}
        assert min(ratios['hf'])>1.05,'No reproducible whole-step improvement over HF'
        results[seq]=dict(step_ms={v:statistics.median(rows[(i,v)]['step_ms'] for i in range(3))
            for v in ('hf','compile','hf_adamw')},optimizer_ms={v:statistics.median(rows[(i,v)]['optimizer_ms']
            for i in range(3)) for v in ('hf','compile','hf_adamw')},paired_speedups=ratios,
            max_peak_gib={v:max(rows[(i,v)]['peak_allocated_bytes'] for i in range(3))/2**30
            for v in ('hf','compile','hf_adamw')})
    trajectory=json.loads(Path('artifacts/triton_adamw_rounded_trajectory_v2.json').read_text())
    candidate=trajectory['variants']['triton_adamw'];assert candidate['accepted']
    assert trajectory['steps']==64 and trajectory['seq']==512
    report=dict(scope='Complete Qwen3-1.7B training step, BF16 parameters/moments; native forward/backward unchanged',
        performance=results,trajectory={k:v for k,v in candidate.items() if k not in ('train_losses','heldout_losses')},
        update_validation=junit('artifacts/triton_adamw_rounding_validation_v3.xml'),
        api_validation=junit('artifacts/triton_adamw_api_validation_v4.xml'),
        regression=junit('artifacts/adamw_contribution_regression_v1.xml'),
        limits=['FP32 scalar math differs from native double hyperparameter expressions; not bitwise identical',
                '64-step trajectory only, not convergence or task-accuracy proof',
                'Single GPU, batch1, no checkpointing; no AMP scaling/AMSGrad/distributed guarantee',
                'Three randomized process rounds, no cross-day or locked-clock claim'],
        conclusion='Verified prototype contribution against standard HF training; no universal best-training-stack claim')
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
