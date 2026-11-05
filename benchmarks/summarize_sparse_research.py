"""Read-only acceptance calculation plus a NEW, non-overwriting audit artifact.

Never changes model defaults. Hashing the checkpoint happens at final audit,
not retroactively described as having happened at benchmark start.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from benchmarks.benchmark_frequency_sparse import ROOT,MODEL,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/sparse_research_audit.json')
    parser.add_argument('--junit',type=Path,default=ROOT/'artifacts/sparse_research_regression.xml')
    args=parser.parse_args()
    rows=[]
    for filename in ('frequency_sparse_pilot.json','frequency_sparse_long.json','block_skip_e2e_confirm.json'):
        report=json.loads((ROOT/'artifacts'/filename).read_text())
        for case in report['cases']:
            for name,q in case['quality'].items():
                speed=case['timing']['paired'][name]['pipeline_ms']
                quality_pass=(q['finite'] and q['logits_relative_l2_max']<.02 and q['top1_agreement']>=.99
                              and q['kl_mean']<=.005 and q['continuation_ppl_ratio']<=1.01)
                speed_pass=speed['paired_median_speedup']>=1.05 and speed['bootstrap_95ci'][0]>1
                rows.append(dict(artifact=filename,seq=case.get('seq_len',case.get('prompt_tokens')),
                    candidate=name,local_quality_pass=quality_pass,speed_pass=speed_pass,
                    end_to_end_paired_speedup=speed['paired_median_speedup'],speed_ci=speed['bootstrap_95ci'],
                    promoted=False,reason='Quality and speed gates must BOTH pass; external tasks additionally required.'))
    initial=json.loads((ROOT/'artifacts/frequency_sparse_pilot.json').read_text())['manifest']['source_sha256']
    core=['model/hf_inference.py','model/static_decode.py']
    baseline_unchanged={p:sha(ROOT/p)==initial[p] for p in core}
    if not all(baseline_unchanged.values()):raise AssertionError('Default source changed during experiment')
    xml=ET.parse(args.junit).getroot()
    suites=[xml] if xml.tag=='testsuite' else xml.findall('testsuite')
    tests={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
    audit=dict(decision='Do not promote either approximate method; keep default dense/Flash path.',
        baseline_source_unchanged=baseline_unchanged,acceptance=rows,regression=tests,
        checkpoint_fingerprint_at_final_audit={p.name:sha(p) for p in sorted(MODEL.iterdir())
            if p.is_file() and (p.suffix=='.safetensors' or p.name in
                ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'))},
        not_attempted=['KVarN low-bit KV','DFlash target-specific draft model/training'],
        limitations=['Small WikiText windows, not task-quality evaluation',
                     'single GPU, no serving throughput, no cross-day repetitions',
                     'Fixed B=1 and uniform non-padded context',
                     'No default/runtime dependency/checkpoint modifications'])
    with args.output.open('x') as f:json.dump(audit,f,indent=2)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
