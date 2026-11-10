"""Quality follow-up selected by repeated throughput, never by test loss."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.summarize_fsdp_shape_confirmation import summarize
from scripts.freeze_fsdp_reference import verify_frozen
from training.fsdp_provenance import source_snapshot


def select_quality_policy(report):
    policies=report['spec']['policies']
    if set(policies)!={'interval_k35','all_k32','all_k36'}:
        raise ValueError('Unexpected candidate set')
    times={r['case']:r['median_step_seconds'] for r in report['records']}
    baseline='interval_k35'
    eligible=[baseline]
    ratios={}
    for name in ('all_k32','all_k36'):
        ratios[name]=[times[f'r{r}_{baseline}']/times[f'r{r}_{name}'] for r in (0,1)]
        if all(x>1.02 for x in ratios[name]):eligible.append(name)
    # Preregistered fallback: retain the existing S512 policy unless a new one
    # exceeds 2% in BOTH orders. This is not a significance claim.
    chosen=min(eligible,key=lambda n:sum(times[f'r{r}_{n}'] for r in (0,1)))
    return dict(candidate=chosen,policy=policies[chosen],throughput_ratios=ratios,
                selection_uses_test_loss=False,scope='Repeated S512 performance selection only')


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume-pid',type=int,required=True)
    p.add_argument('--resume-queue',type=Path,required=True)
    p.add_argument('--performance',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**fields):
        (a.output/'status.json').write_text(json.dumps(dict(phase=phase,**fields),indent=2))
        print(phase,fields,flush=True)
    try:
        status('waiting_for_resume_queue')
        while True:
            try:cmd=Path(f'/proc/{a.resume_pid}/cmdline').read_bytes()
            except FileNotFoundError:break
            if b'scripts.queue_real_fsdp_resume' not in cmd:break
            time.sleep(15)
        resume=json.loads((a.resume_queue/'status.json').read_text())
        # A storage guard rejection does not invalidate training, but a replay
        # or predecessor failure must never silently authorize the quality run.
        if resume['phase'] not in ('finished','resource_guard_rejected'):
            raise ValueError('Resume queue did not reach an acceptable terminal state')
        spec=json.loads((a.performance/'spec.json').read_text())
        report=summarize(a.performance,spec)
        selection=select_quality_policy(report)
        (a.output/'selection.json').write_text(json.dumps(selection,indent=2))
        current_sources=source_snapshot(Path.cwd())
        first=json.loads((a.performance/report['records'][0]['case']/'result.json').read_text())
        if current_sources!=first['manifest']['implementation_sources']:
            raise ValueError('Training source differs from confirmed performance source')
        frozen=json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
        status('verifying_frozen_inputs')
        verify_frozen(frozen)
        (a.output/'reference.json').write_text(json.dumps(dict(content_sha256=frozen['content_sha256'],
            implementation_sources=current_sources,scope='Frozen content verified before quality jobs'),indent=2))
        keep,every,mode=selection['policy']
        cases=[('baseline','official_keep24',[]),('candidate','official',
            ['--keep-full-layers',str(keep),'--checkpoint-every',str(every),'--checkpoint-mode',mode])]
        for name,backend,flags in cases:
            status('running_quality_campaign',case=name,backend=backend,flags=flags)
            with (a.output/(name+'.log')).open('x') as stream:
                subprocess.run([sys.executable,'-m','scripts.run_fsdp_convergence','--backend',backend,
                    '--run-tiny','--launch-long','--output',str(a.output/name),*flags],
                    stdout=stream,stderr=subprocess.STDOUT,check=True)
            raw=json.loads((a.output/name/f'{backend}_full_protocol/result.json').read_text())
            if not raw.get('converged') or raw.get('stop_reason')!='validation_plateau':
                raise ValueError(f'{name} stopped without convergence')
            if raw.get('source_snapshot')!=current_sources:
                raise ValueError('Quality result source differs from selected performance source')
        status('verifying_frozen_inputs_after_training')
        verify_frozen(frozen)
        status('finished',scope='Both full quality jobs completed; independently audit before claiming benefit')
    except Exception as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
