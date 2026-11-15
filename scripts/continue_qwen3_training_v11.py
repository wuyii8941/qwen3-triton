"""Finish the live screen, explore larger retention, then run full controls."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

from qwen3_triton.gradient_archive import sha

ROOT = Path(__file__).resolve().parents[1]


def higher_profiles(control, winner):
    def row(name, keep, checkpoint=1, plan=True, head=False):
        return dict(name=name,keep_full_layers=keep,checkpoint_every=checkpoint,
                    skip_embedding_backward_gather=True,retain_lm_head=head,gemm_plan=plan)
    return [control,dict(winner,name='incumbent'),row('retain24_plan',24),
            row('retain36',36,plan=False),row('retain36_plan',36),
            row('retain36_head_plan',36,head=True),row('retain36_cp2_plan',36,checkpoint=2),
            row('retain24_cp0_plan',24,checkpoint=0),row('retain36_cp0_plan',36,checkpoint=0)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-screen',type=Path,required=True)
    parser.add_argument('--initial-pid',type=int,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args = parser.parse_args()
    assert args.initial_pid > 0
    initial, base = args.initial_screen.resolve(), args.output_root.resolve()
    assert base.is_dir() and not base.is_relative_to(Path('/home/tzh'))
    def state(stage, **kwargs):
        value = dict(pid=os.getpid(),stage=stage,**kwargs)
        (base/'pipeline_state.json').write_text(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    state('waiting_initial_screen',initial_pid=args.initial_pid)
    while True:
        status = json.loads((initial/'state.json').read_text())
        if status['stage'] == 'complete_screen_only':
            break
        try:
            os.kill(args.initial_pid,0)
        except ProcessLookupError as exc:
            raise RuntimeError('Initial controller terminated before completing; inspect its existing logs') from exc
        proc = Path(f'/proc/{args.initial_pid}/stat')
        if proc.exists():
            assert proc.read_text().split()[2] != 'Z', 'Initial controller is terminal without completion'
            command = Path(f'/proc/{args.initial_pid}/cmdline').read_bytes().split(b'\0')
            assert str(initial).encode() in command, 'Initial PID no longer belongs to this screen'
        time.sleep(5)
    initial_result = json.loads((initial/'result.json').read_text())
    initial_manifest = json.loads((initial/'manifest.json').read_text())
    assert initial_result['complete']
    winner = initial_result['selected_for_formal_comparison']
    assert winner is not None, 'No eligible initial strategy; continue research separately'
    profiles = dict(parent_result_sha256=sha(initial/'result.json'),
        planner_source=str(Path(__file__)),planner_source_sha256=sha(Path(__file__)),
        reason='Initial screen shows useful retention gains while allocated peak remains near34GiB. Retest incumbent and test24/36layers plus GEMM/checkpoint/head choices under the same cap.',
        roles=higher_profiles(initial_manifest['roles'][0],winner))
    profile_path = base/'higher_retention_profiles_001.json'
    with profile_path.open('x') as stream:
        stream.write(json.dumps(profiles,indent=2)+'\n')
    followup = base/'retention_screen_001'
    state('running_higher_retention_screen',output=str(followup))
    subprocess.run(['bash',str(ROOT/'scripts/research_python.sh'),str(ROOT/'scripts/screen_qwen3_retention_v11.py'),
        '--parent-screen',str(initial),'--profiles',str(profile_path),'--output',str(followup)],cwd=ROOT,check=True)
    result = json.loads((followup/'result.json').read_text())
    assert result['complete'] and result['selected_for_formal_comparison'] is not None
    formal = base/'formal_001'
    state('running_formal_comparison',output=str(formal))
    subprocess.run(['bash',str(ROOT/'scripts/research_python.sh'),str(ROOT/'scripts/run_qwen3_training_v11.py'),
        '--screen',str(followup),'--output',str(formal)],cwd=ROOT,check=True)
    review = json.loads((formal/'independent_review.json').read_text())
    assert review['complete'] and review['audit_passed']
    state('complete_reviewed',qualified_training_improvement=review['qualified_training_improvement'])


if __name__ == '__main__':
    main()
