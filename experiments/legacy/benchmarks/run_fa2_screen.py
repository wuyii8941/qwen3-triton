"""Isolated FA2 screen: six operator shapes and two full-model step cases."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def gpu_snapshot():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    return result.stdout.strip()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    root=Path('artifacts')/f'fa2_screen_{args.tag}';root.mkdir(exist_ok=False)
    cases=[dict(mode='operator',seq=seq,dtype=dtype,batch=batch)
           for batch,seq in ((1,512),(1,2048),(2,512)) for dtype in ('bf16','fp16')]
    cases += [dict(mode='model',seq=seq,dtype='bf16',batch=1) for seq in (512,2048)]
    summary=[]
    for case in cases:
        name=f'{case["mode"]}_b{case["batch"]}_s{case["seq"]}_{case["dtype"]}'
        output=root/f'{name}.json';before=gpu_snapshot()
        command=[sys.executable,'-m','benchmarks.benchmark_fa2_comparison',
                 '--output',str(output),'--iterations','30' if case['mode']=='operator' else '12']
        for key,value in case.items():command += [f'--{key}',str(value)]
        print('START',name,flush=True)
        with (root/f'{name}.log').open('x') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        stage=dict(case=case,returncode=completed.returncode,gpu_before=before,gpu_after=gpu_snapshot())
        with (root/f'{name}.stage.json').open('x') as out:json.dump(stage,out,indent=2)
        if completed.returncode:
            raise RuntimeError(f'{name} failed; inspect {root/(name+".log")}')
        result=json.loads(output.read_text())['result']
        if case['mode']=='operator':
            sdpa=[r['forward_backward']['median_ms'] for r in result['timings'] if r['backend']=='sdpa_hf_gqa']
            fa2=[r['forward_backward']['median_ms'] for r in result['timings'] if r['backend']=='fa2_compact_gqa']
        else:
            sdpa=[r['median_step_ms'] for r in result['trials'] if r['backend']=='sdpa']
            fa2=[r['median_step_ms'] for r in result['trials'] if r['backend']=='flash_attention_2']
        row=dict(case=case,sdpa_ms=statistics.median(sdpa),fa2_ms=statistics.median(fa2),
                 speedup=statistics.median(sdpa)/statistics.median(fa2),checks=result['checks'])
        summary.append(row)
        temp=root/'summary.tmp';temp.write_text(json.dumps(summary,indent=2));temp.replace(root/'summary.json')
        print('END',name,'speedup',round(row['speedup'],4),flush=True)
    print('ALL_SCREEN_CASES_COMPLETE',str(root),flush=True)


if __name__=='__main__':main()
