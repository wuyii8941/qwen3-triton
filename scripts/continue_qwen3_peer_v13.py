"""Await one live full controller, then run the independent review on CPU.

Never restarts training, never treats a timeout as failure, and refuses source
drift. This leaves the GPU lease free throughout CPU reference/archive review.
"""
import argparse,json,os,subprocess,time
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from run_vocab_resource_queue_v12 import dump


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();receipt=a.receipt.resolve()
    assert out.is_relative_to(Path('/data1/tzh')) and receipt.is_relative_to(Path('/data1/tzh')) and not receipt.exists()
    initial=json.loads((out/'state.json').read_text());pid=initial['pid'];proc=Path(f'/proc/{pid}/stat')
    fields=proc.read_text().rsplit(') ',1)[1].split() if proc.exists() else None
    start=fields[19] if fields is not None else None
    reviewer=root/'scripts/review_qwen3_peer_full_v13.py';digest=sha(reviewer)
    dump(receipt,dict(pid=os.getpid(),stage='waiting_specific_full_controller',controller_pid=pid,controller_start_ticks=start,
        full_manifest_sha256=sha(out/'manifest.json'),reviewer_sha256=digest,orchestrator_sha256=sha(Path(__file__))))
    while True:
        state=json.loads((out/'state.json').read_text())
        if state['stage']=='GPU_complete_independent_CPU_review_pending':break
        fields=proc.read_text().rsplit(') ',1)[1].split() if proc.exists() else None
        if fields is None or fields[0] in ('Z','X') or fields[19]!=start:
            raise RuntimeError('Specific full controller ended before GPU-complete state; no automatic restart')
        time.sleep(5)
    assert sha(reviewer)==digest
    record=json.loads(receipt.read_text());record['stage']='CPU_reference_and_full_cohort_review';dump(receipt,record)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run(['bash',str(root/'scripts/research_python.sh'),str(reviewer),'--output',str(out)],cwd=root,env=env,check=True)
    assert sha(reviewer)==digest
    result=json.loads((out/'independent_review.json').read_text());assert result['complete'] and result['audit_passed']
    record.update(stage='CPU_review_complete',qualified_training_improvement=result['qualified_training_improvement'],review_sha256=sha(out/'independent_review.json'))
    dump(receipt,record);print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
