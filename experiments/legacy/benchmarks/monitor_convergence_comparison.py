"""Read-only file snapshot. Does not infer liveness from stale logs."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    report={'root':str(a.root),'liveness':'File snapshot only; check the running process/session separately','jobs':{}}
    for name in ('triton_preflight','calibration','native','triton'):
        directory=a.root/name
        if not directory.exists():continue
        events=directory/'events.jsonl';rows=[]
        if events.exists():
            for line in events.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:continue # Writer may be mid-line.
        summary=directory/'summary.json'
        report['jobs'][name]={'last_event':rows[-1] if rows else None,
            'latest_validation':next((r for r in reversed(rows) if r.get('event') in ('validation','initial_validation')),None),
            'checkpoint_exists':(directory/'last.pt').exists(),
            'terminal_summary':json.loads(summary.read_text()) if summary.exists() else None}
    comparison=a.root/'comparison.json'
    report['comparison']=json.loads(comparison.read_text()) if comparison.exists() else None
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
