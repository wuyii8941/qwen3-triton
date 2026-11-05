"""External process-lifetime clock, without changing frozen training code.

Live event timestamps are observer-read times (poll error <= poll interval
plus OS scheduling). Existing events at attach are marked historical and
must NOT be used as exact wall-time crossings. Includes Python imports and
process teardown, unlike the job's main-entry elapsed clock.
"""
import argparse
import json
import os
from pathlib import Path
import time
import psutil


def find_job(output):
    for process in psutil.process_iter(['pid','cmdline','uids']):
        try:
            if process.info['uids'].real!=os.getuid():continue
            args=process.info['cmdline'] or []
            if 'training.convergence_job' not in args or '--output' not in args:continue
            if Path(args[args.index('--output')+1]).resolve()==output.resolve():return process
        except (psutil.Error,IndexError,OSError):continue


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--poll',type=float,default=.25);a=p.parse_args()
    clock_path=a.root/'external_process_clocks.json'
    clocks=json.loads(clock_path.read_text()) if clock_path.exists() else {}
    for name in ('native','triton'):
        if name in clocks:continue
        directory=a.root/name;wait_start=time.monotonic();process=None
        while process is None:
            process=find_job(directory)
            if process is not None:break
            if (directory/'summary.json').exists():raise RuntimeError(f'{name} already exited; cannot observe retrospectively')
            if time.monotonic()-wait_start>120:raise RuntimeError(f'No live {name} process after120s')
            time.sleep(a.poll)
        start=process.create_time();attached=time.time()
        # Process discovery precedes argparse/output-directory creation.
        # Do not create the directory ourselves: the job rejects existing output.
        while not directory.exists():
            if not process.is_running():raise RuntimeError(f'{name} exited before creating output')
            time.sleep(a.poll)
        path=directory/'events.jsonl';offset=0
        if path.exists():offset=path.stat().st_size
        destination=directory/'external_clock_events.jsonl'
        with destination.open('x') as out:
            out.write(json.dumps(dict(event='observer_attached',pid=process.pid,process_start_epoch=start,
                attached_epoch=attached,existing_event_bytes=offset,poll_seconds=a.poll))+'\n');out.flush()
            pending=''
            while True:
                try:
                    alive=process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    alive=False
                if not alive:break
                if path.exists():
                    with path.open() as source:source.seek(offset);new=source.read();offset=source.tell()
                    pending+=new
                    while '\n' in pending:
                        line,pending=pending.split('\n',1)
                        if not line:continue
                        event=json.loads(line);observed=time.time()
                        out.write(json.dumps(dict(observed_epoch=observed,process_elapsed_seconds=observed-start,
                                                  historical=False,job_event=event))+'\n');out.flush()
                time.sleep(a.poll)
            end=time.time()
        clocks[name]=dict(pid=process.pid,process_start_epoch=start,exit_observed_epoch=end,
            process_lifetime_seconds=end-start,attached_late_seconds=attached-start,
            poll_seconds=a.poll,summary_present=(directory/'summary.json').exists())
        (a.root/'external_process_clocks.json').write_text(json.dumps(clocks,indent=2))
        print(name,clocks[name],flush=True)


if __name__=='__main__':main()
