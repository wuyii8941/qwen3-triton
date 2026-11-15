"""Wait for one exact screen process, then run the frozen full comparison.

Never restart a live or failed screen. A terminal successful screen result and
its current source hashes are required by the full controller before any GPU
admission. This process creates no dense gradient archives or new weights.
"""
from pathlib import Path
import argparse,json,os,subprocess,time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--screen-pid',type=int,required=True);p.add_argument('--checkpoint-screen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--state-file',type=Path,required=True)
    args=p.parse_args()
    assert args.state_file.resolve().is_relative_to(Path('/data1/tzh')) and not args.state_file.exists()
    assert args.output.resolve().is_relative_to(Path('/data1/tzh')) and not args.output.exists()
    def state(stage,**extra):
        a=dict(pid=os.getpid(),stage=stage,**extra);args.state_file.write_text(json.dumps(a)+'\n');print(json.dumps(a),flush=True)
    state('waiting_exact_screen_process',screen_pid=args.screen_pid)
    proc=Path(f'/proc/{args.screen_pid}/stat')
    initial=proc.read_text().split()[21] if proc.exists() else None
    while proc.exists():
        fields=proc.read_text().split()
        if fields[2]=='Z' or fields[21]!=initial:break
        time.sleep(5)
    receipt=json.loads((args.screen/'result.json').read_text())
    assert receipt['complete'] and receipt['eligible_for_formal']
    assert all(r['trace'] and r['trace']['passed'] for r in receipt['results'].values())
    state('running_full_comparison')
    subprocess.run(['bash',str(ROOT/'scripts/research_python.sh'),str(ROOT/'scripts/run_vocab_full_v12.py'),
        '--screen',str(args.screen),'--checkpoint-screen',str(args.checkpoint_screen),'--output',str(args.output)],cwd=ROOT,check=True)
    report=json.loads((args.output/'independent_review.json').read_text())
    assert report['complete'] and report['audit_passed']
    state('complete_reviewed',qualified_training_improvement=report['qualified_training_improvement'])


if __name__=='__main__':main()
