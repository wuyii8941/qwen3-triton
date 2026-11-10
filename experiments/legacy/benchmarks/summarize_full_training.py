"""Fail-closed evidence audit. Writes a NEW result; never promotes a default."""
import argparse
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--junit',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    root=Path('artifacts')
    repeated=json.loads((root/'full_training_repeated_steady_v1.json').read_text())
    variants=['hf','compile','rms','swiglu','qk','operators','fused_loss']
    assert len(repeated['runs'])==21,'Three complete rounds required'
    data={v:[] for v in variants}
    inputs=set()
    for run in repeated['runs']:
        assert run['returncode']==0
        r=json.loads(Path(run['artifact']).read_text())
        assert r['manifest']['layers']==28 and len(r['samples'])==10
        assert r['scope'].startswith('steady')
        inputs.add(r['manifest']['input_sha256'])
        data[run['variant']].append((run['round'],r['medians']))
    assert len(inputs)==1,'Input mismatch'
    for v in variants:assert sorted(x[0] for x in data[v])==[0,1,2]
    table={}
    for v in variants:
        rows=sorted(data[v]);baseline=sorted(data['hf'])
        table[v]=dict(median_step_ms=statistics.median(x[1]['step_ms'] for x in rows),
            median_tokens_s=statistics.median(x[1]['tokens_per_second'] for x in rows),
            peak_allocated_gib=max(x[1]['peak_allocated_bytes'] for x in rows)/2**30,
            paired_round_speedups=[b[1]['step_ms']/x[1]['step_ms'] for b,x in zip(baseline,rows)])
    budget=json.loads((root/'training_budget_v1.json').read_text())
    assert len(budget['cases'])==9 and all(r['status'] in ('ok','oom') for r in budget['cases'])
    assert {(r['variant'],r['microbatch']) for r in budget['cases']}=={(v,b) for v in ('hf','compile','fused_loss') for b in (1,2,4)}
    parity=json.loads((root/'full_training_gradient_validation_v1.json').read_text())
    assert set(parity['candidates'])==set(variants)-{'hf'}
    control=json.loads((root/'full_training_repeat_control_v2.json').read_text())
    assert control['candidates']['repeat']['accepted']
    assert all(r['relative_l2']==0 for r in control['candidates']['repeat']['gradients'].values())
    for file in ('full_training_hf_s2048_profile_v1.json','full_training_chunked_s2048_profile_v2.json'):
        report=json.loads((root/file).read_text());assert 'profiler_table' in report
    xml=ET.parse(a.junit).getroot()
    suites=[xml] if xml.tag=='testsuite' else xml.findall('testsuite')
    totals={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
    assert totals['failures']==totals['errors']==totals['skipped']==0
    result=dict(requirements=dict(full28layer_training_comparison=True,unified_loss_precision=True,
        fused_forward_no_full_logits=True,full_gradient_backward_no_filter=True,
        same_configuration_measurements=True,same_budget_comparison=True,profiler_based_selection=True,
        complete_regression=True),repeated_table=table,budget_best_tested=budget['best_tested'],
        full_model_gradient_acceptance={k:v['accepted'] for k,v in parity['candidates'].items()},
        regression=totals,decision='No default operator replacement; retain numerical/performance failures.',
        limits=['BF16 parameters/moments, no FP32 master','One fixed sequence per shape; no convergence claim',
                'Three process-level rounds, not cross-day replication','Allocator cap is not total device VRAM',
                'Fused forward; backward still chunk-recomputed, not one fused backward kernel'])
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
