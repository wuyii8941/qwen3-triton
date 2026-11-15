"""Audit first-clean selection and preserve the original numerical reviewer."""
from pathlib import Path
import argparse,json,os,subprocess
from qwen3_triton.gradient_archive import sha


def audit_resources(out):
    manifest=json.loads((out/'manifest.json').read_text());history=json.loads((out/'attempts.json').read_text())
    roles=[r['name'] for r in manifest['roles']];position=0;accepted=[];rejected=[]
    for number,row in enumerate(history,1):
        assert row['number']==number and position<len(roles) and row['role']==roles[position]
        folder=out/row['folder'];assert folder.resolve().is_relative_to(out.resolve())
        assert sha(folder/'monitor.json')==row['monitor_sha256'] and sha(folder/'worker.log')==row['log_sha256']
        monitor=json.loads((folder/'monitor.json').read_text());owned=set(monitor['known_owned_pids'])
        for sample in monitor['samples']:
            if 'observation_error' not in sample:
                assert sample['foreign_pids']==[p['pid'] for p in sample['apps'] if p['pid'] not in owned]
        assert monitor['foreign_process_seen']==any(s.get('foreign_pids') for s in monitor['samples'])
        assert monitor['observation_errors']==sum('observation_error' in s for s in monitor['samples'])
        dirty=bool(monitor['foreign_process_seen'] or monitor['observation_errors'])
        assert dirty==row['resource_contaminated'] and monitor['returncode']==row['returncode']
        assert monitor['source_snapshot']==manifest['core_snapshot'] and monitor['samples']
        if not row['accepted']:
            # A clean successful attempt may never be skipped for a faster one.
            assert dirty
            rejected.append(dict(number=number,role=row['role'],monitor_sha256=row['monitor_sha256'],reason=monitor['cancelled_for_resource']))
            continue
        assert not dirty and monitor['returncode']==0
        assert all(not s.get('foreign_pids') and 'observation_error' not in s and
                   all(p['pid'] in owned for p in s['apps']) for s in monitor['samples'])
        role=row['role'];result=folder/'run/result.json';trace=folder/'run/training_trace.json'
        assert sha(result)==row['result_sha256'] and sha(trace)==row['trace_sha256']
        a=json.loads(result.read_text());assert a['complete'] and not a['screen_only']
        assert all(r['updates']==27 and r['parameter_count']==r['optimizer_state_count']==399 for r in a['ranks'])
        assert (out/role).is_symlink() and (out/role).resolve()==(folder/'run').resolve()
        assert (out/(role+'.gpu.json')).resolve()==(folder/'monitor.json').resolve()
        assert (out/(role+'.log')).resolve()==(folder/'worker.log').resolve()
        accepted.append(dict(role=role,attempt=number,result_sha256=row['result_sha256'],trace_sha256=row['trace_sha256'],monitor_sha256=row['monitor_sha256']))
        position+=1
    assert position==len(roles)
    return dict(complete=True,first_clean_attempt_policy_passed=True,fixed_role_order_passed=True,
        accepted=accepted,rejected=rejected,no_timing_based_attempt_selection=True,
        manifest_sha256=sha(out/'manifest.json'),attempt_history_sha256=sha(out/'attempts.json'),
        reviewer_sha256=sha(Path(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();assert out.is_relative_to(Path('/data1/tzh'))
    report=audit_resources(out)
    with (out/'resource_review.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    root=Path(__file__).resolve().parents[1];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run(['bash',str(root/'scripts/research_python.sh'),str(root/'scripts/review_vocab_full_v12.py'),
                    '--output',str(out)],cwd=root,env=env,check=True)


if __name__=='__main__':main()
