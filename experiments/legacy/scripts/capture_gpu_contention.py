"""Point-in-time read-only GPU ownership/activity observation, not attribution."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess


def descendants(parent,rows):
    owned={parent}
    while True:
        expanded=owned|{pid for pid,ppid in rows if ppid in owned}
        if expanded==owned:return owned
        owned=expanded


def parse_pmon(text,owned):
    rows=[]
    for line in text.splitlines():
        fields=line.split()
        if len(fields)<4 or not fields[0].isdigit() or not fields[1].isdigit():continue
        rows.append(dict(gpu=int(fields[0]),pid=int(fields[1]),
            sm_percent=int(fields[3]) if fields[3].isdigit() else None,
            belongs_to_experiment=int(fields[1]) in owned))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent-pid',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    # Do not print arbitrary process command lines or environment variables.
    parent=Path(f'/proc/{args.parent_pid}')
    cmd=parent.joinpath('cmdline').read_bytes().split(b'\0')
    allowed={b'scripts.screen_fsdp_exchange_40',b'scripts.confirm_fsdp_exchange'}
    if not any(cmd[i]==b'-m' and cmd[i+1] in allowed for i in range(len(cmd)-1)):
        raise ValueError('A live known experiment supervisor is required')
    if parent.joinpath('cwd').resolve()!=Path.cwd():raise ValueError('Wrong workspace')
    started=datetime.now(timezone.utc).isoformat()
    table=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True)
    owned=descendants(args.parent_pid,[tuple(map(int,line.split())) for line in table.splitlines()])
    pmon=subprocess.check_output(['nvidia-smi','pmon','-s','um','-c','1'],text=True)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu,clocks.sm,power.draw',
        '--format=csv,noheader'],text=True)
    rows=parse_pmon(pmon,owned)
    report=dict(sample_started_utc=started,sample_finished_utc=datetime.now(timezone.utc).isoformat(),
        parent_pid=args.parent_pid,owned_pid_snapshot=sorted(owned),process_samples=rows,
        external_nonzero_sm=[r for r in rows if not r['belongs_to_experiment'] and r['sm_percent'] is not None and r['sm_percent']>0],
        raw_pmon=pmon,raw_gpu=gpu,
        scope='One process-tree/PMON/NVML observation; samples are not simultaneous, unknown is not zero. No GPU writes, no full-run attribution, no utilization-to-latency correction or causal protocol conclusion.')
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(dict(owned_pid_snapshot=report['owned_pid_snapshot'],
        external_nonzero_sm=report['external_nonzero_sm'],scope=report['scope']),indent=2))


if __name__=='__main__':main()
