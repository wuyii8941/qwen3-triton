"""Recompute module numerical/timing checks and selected-GPU ownership evidence.

Module latency below means the sum of separate forward/backward medians.
It excludes FSDP/NCCL/AdamW and does not establish a whole-training gain.
"""
from pathlib import Path
import argparse,json,math,statistics
from qwen3_triton.gradient_archive import sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Completed module controller directory')
    parser.add_argument('--source-root',type=Path,default=ROOT)
    args=parser.parse_args();out=args.output.resolve();source=args.source_root.resolve()
    assert not out.is_relative_to(Path('/home/tzh'))
    result_path=out/'measurements/result.json';monitor_path=out/'gpu.json'
    result=json.loads(result_path.read_text());monitor=json.loads(monitor_path.read_text())
    assert result['complete'] and len(result['rows'])==49
    assert monitor['returncode']==0 and monitor['selected_device_clean']
    owned=set(monitor['owned_pids']);uuid=monitor['gpu_uuid']
    assert monitor['samples']
    for sample in monitor['samples']:
        assert 'observation_error' not in sample
        foreign=[a['pid'] for a in sample['apps'] if a['gpu_uuid']==uuid and a['pid'] not in owned]
        assert not foreign and not sample['foreign_selected_pids']
    for name,h in result['sources'].items():assert sha(source/name)==h,name
    checks=[r for r in result['rows'] if 'numeric' in r]
    timed=[r for r in result['rows'] if 'order' in r]
    assert len(checks)==7 and len(timed)==42
    for row in checks:
        assert row['local_VJP_passed'] and row['comparison']=='v10_vs_v9'
        for check in row['numeric'].values():
            assert check['finite'] and all(math.isfinite(check[k]) and check[k]>=0 for k in ('absolute_error_norm','reference_norm','relative_l2','max_abs','threshold'))
            threshold=.001*max(check['reference_norm'],1.e-8)+1.e-6
            assert threshold==check['threshold']
            assert check['passed']==(check['absolute_error_norm']<threshold) and check['passed']
    for row in timed:
        assert len(row['forward_ms'])==len(row['backward_ms'])==20
        assert all(math.isfinite(v) and v>0 for v in row['forward_ms']+row['backward_ms'])
        assert statistics.median(row['forward_ms'])==row['median_forward_ms']
        assert statistics.median(row['backward_ms'])==row['median_backward_ms']
    groups=[]
    for layer,kind in [(i,k) for i in (0,17,35) for k in ('attention','mlp')]+[(None,'lm_head')]:
        times={}
        for variant in ('hf','v9','v10'):
            selected=[r for r in timed if r['layer']==layer and r['module']==kind and r['variant']==variant]
            assert [r['order'] for r in selected]==[1,2]
            times[variant]=[r['median_forward_ms']+r['median_backward_ms'] for r in selected]
        groups.append(dict(layer=layer,module=kind,times_ms=times,
            reduction_vs_V9=[1-a/b for a,b in zip(times['v10'],times['v9'])],
            reduction_vs_HF=[1-a/b for a,b in zip(times['v10'],times['hf'])]))
    report=dict(complete=True,audit_passed=True,source_root=str(source),plan=result['plan'],
        selected_device_clean=True,local_VJP_groups=7,timed_cases=42,
        maximum_relative_l2_vs_V9=max(c['relative_l2'] for r in checks for c in r['numeric'].values()),
        groups=groups,whole_training_numerically_qualified=False,
        sources={str(p):sha(p) for p in (Path(__file__),result_path,monitor_path)},scope=__doc__)
    with (out/'independent_review.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,audit_passed=True,groups=groups),indent=2),flush=True)


if __name__=='__main__':main()
