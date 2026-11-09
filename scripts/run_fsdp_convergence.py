"""Gate long training behind serial tiny/real preflights; data-disk logs only."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predecessor',type=int,default=0,help='Optional live tiny-preflight PID')
    p.add_argument('--backend',choices=['official','official_keep8','official_keep24'],default='official')
    p.add_argument('--keep-full-layers',type=int)
    p.add_argument('--checkpoint-every',type=int,default=0)
    p.add_argument('--checkpoint-mode',choices=['full','selective'],default='full')
    p.add_argument('--run-tiny',action='store_true',help='Run matching tiny preflight once GPUs are idle')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--launch-long',action='store_true')
    p.add_argument('--tiny-result',type=Path,default=Path('artifacts/fsdp_convergence_tiny_official_v1/result.json'))
    args=p.parse_args()
    from training.fsdp_convergence_policy import resolve_policy
    policy=resolve_policy(args.backend,args.keep_full_layers,args.checkpoint_every,args.checkpoint_mode)
    policy_cli=['--checkpoint-every',str(args.checkpoint_every),'--checkpoint-mode',args.checkpoint_mode]
    if args.keep_full_layers is not None:policy_cli+=['--keep-full-layers',str(args.keep_full_layers)]
    args.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**extra):
        (args.output/'status.json').write_text(json.dumps(dict(phase=phase,updated=time.time(),**extra),indent=2))
        print(phase,flush=True)
    try:
        def idle(name):
            status('waiting_for_idle_gpus',next_case=name)
            while True:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True).splitlines()
                readings=[tuple(int(v.strip()) for v in row.split(',')) for row in rows]
                if len(readings)>=4 and all(memory<2048 and utilization<10 for memory,utilization in readings[:4]):return
                time.sleep(15)
        status('waiting_for_tiny_preflight')
        command=Path(f'/proc/{args.predecessor}/cmdline')
        while command.exists():
            try:text=command.read_bytes().replace(b'\0',b' ')
            except FileNotFoundError:break
            if b'fsdp_convergence_job.py' not in text:break
            time.sleep(10)
        if args.run_tiny:
            idle('tiny_preflight');status('tiny_preflight')
            with (args.output/'tiny_preflight.log').open('x') as stream:
                subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                    'training/fsdp_convergence_job.py','--backend',args.backend,'--tiny','--preflight-steps','2',
                    '--output',str(args.output/'tiny_preflight'),*policy_cli],stdout=stream,stderr=subprocess.STDOUT,check=True)
            args.tiny_result=args.output/'tiny_preflight/result.json'
        tiny=json.loads(args.tiny_result.read_text())
        if tiny['backend']!=args.backend:raise ValueError('Tiny preflight backend mismatch')
        tiny_policy=tiny.get('optimization')
        if tiny_policy is None:
            if args.keep_full_layers is not None or args.checkpoint_every or args.checkpoint_mode!='full':
                raise ValueError('Explicit policy requires a matching policy-aware tiny preflight')
        elif any(tiny_policy.get(key)!=policy[key] for key in
                 ('requested_keep_full_layers','checkpoint_every','checkpoint_mode')):
            raise ValueError('Tiny preflight optimization policy mismatch')
        assert tiny['complete'] and not tiny['converged'] and tiny['stop_reason']=='preflight_cutoff'
        assert tiny['best_parameters_restored'] and len(tiny['validation'])==3
        assert all(v['tokens']==17*15 for v in tiny['validation'])
        def run(name,steps,protocol=None):
            idle(name)
            status(name)
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'training/fsdp_convergence_job.py','--backend',args.backend,'--output',str(args.output/name),*policy_cli]
            if steps is not None:cmd+=['--preflight-steps',str(steps)]
            if protocol is not None:cmd+=['--protocol',str(protocol)]
            with (args.output/f'{name}.log').open('w') as stream:
                subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
            return json.loads((args.output/name/'result.json').read_text())
        # Validate again AFTER AdamW states exist. An initial no-grad forward
        # alone cannot detect the later validation memory peak.
        protocol_path=Path('training/fsdp_convergence_protocol_v1.json')
        preflight_protocol=json.loads(protocol_path.read_text())
        preflight_protocol['validation']['every_updates']=2
        preflight_protocol_path=args.output/'real_preflight_protocol.json'
        preflight_protocol_path.write_text(json.dumps(preflight_protocol,indent=2))
        preflight=run('real_8b_preflight',2,preflight_protocol_path)
        assert preflight['complete'] and not preflight['converged'] and preflight['best_parameters_restored']
        if [r['step'] for r in preflight.get('validation',[])]!=[0,2]:
            raise ValueError('Real preflight must validate after initialized optimizer states')
        if preflight.get('protocol')!=preflight_protocol:
            raise ValueError('Real preflight protocol mismatch')
        if args.launch_long:
            result=run(f'{args.backend}_full_protocol',None)
            status('finished',converged=result['converged'],stop_reason=result['stop_reason'])
        else:status('preflights_passed')
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
