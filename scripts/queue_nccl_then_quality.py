"""Bounded communication diagnostic before the already planned quality jobs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.summarize_nccl_protocol import summarize


def bounded_run(command, log, env, timeout=240, grace_seconds=15):
    """A new session owns only this probe; timeout never kills other jobs."""
    with Path(log).open('x') as stream:
        process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:return process.wait(timeout=timeout),False
        except subprocess.TimeoutExpired:
            # The live session leader was created here, not inferred by name.
            if process.poll() is None and os.getpgid(process.pid)==process.pid:
                os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=grace_seconds)
            return process.returncode,True


def large_message_candidate(report):
    if not report.get('complete'):raise ValueError('Complete protocol comparison required')
    flags=report['simple_over_two_percent_in_both_orders']
    return any(flags[f'{size}/{op}'] for size in ('decoder','head')
               for op in ('all_gather','reduce_scatter'))


def is_resume_queue_command(command):
    return any(name in command for name in
               (b'scripts.queue_real_fsdp_resume',b'scripts.queue_existing_fsdp_resume'))


def resources_ready(readings,*,communication=False):
    """Communication's <=~12 GiB staging is not a full-model memory budget.

    Keep >=32 GiB free for operator probes and the original <2 GiB resident
    gate for model jobs. Both require low current utilization; no reservation
    or future-neighbor isolation is implied on this shared machine.
    """
    if len(readings)<4:return False
    return all(util<10 and (free>=32768 if communication else used<2048)
               for used,free,util in readings[:4])


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume-pid',type=int,required=True)
    p.add_argument('--resume-queue',type=Path,required=True)
    p.add_argument('--performance',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exchange-screen',action='store_true',help='Gate BF16 exchange operator before long quality jobs')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**fields):
        (a.output/'status.json').write_text(json.dumps(dict(phase=phase,**fields),indent=2))
        print(phase,fields,flush=True)
    def idle(case,*,communication=False):
        status('waiting_for_idle_gpus',case=case,communication=communication)
        while True:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).splitlines()
            values=[tuple(int(v.strip()) for v in row.split(',')) for row in rows]
            if resources_ready(values,communication=communication):return
            time.sleep(15)
    try:
        status('waiting_for_resume')
        while True:
            try:cmd=Path(f'/proc/{a.resume_pid}/cmdline').read_bytes()
            except FileNotFoundError:break
            if not is_resume_queue_command(cmd):break
            time.sleep(15)
        previous=json.loads((a.resume_queue/'status.json').read_text())
        if previous['phase'] not in ('finished','resource_guard_rejected'):
            raise ValueError('Resume predecessor unsuccessful; resolve its recorded failure before follow-up')
        root=a.output/'protocol';root.mkdir()
        complete=True
        for case,proto in (('r0_auto','auto'),('r0_simple','Simple'),('r1_simple','Simple'),('r1_auto','auto')):
            idle(case,communication=True);status('running_protocol_probe',case=case)
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,TUNING')
            for name in ('NCCL_PROTO','NCCL_ALGO','NCCL_DEBUG_FILE','NCCL_TOPO_DUMP_FILE','NCCL_GRAPH_DUMP_FILE'):
                env.pop(name,None)
            if proto=='Simple':env['NCCL_PROTO']='Simple'
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                'benchmarks/nccl_protocol_probe.py','--expected-proto',proto,'--output',str(root/case)]
            code,timed_out=bounded_run(cmd,root/(case+'.log'),env)
            with (root/'status.log').open('a') as stream:stream.write(f'{case} returncode={code}\n')
            if code or timed_out:
                complete=False
                (root/'failure.json').write_text(json.dumps(dict(case=case,returncode=code,
                    timed_out=timed_out,scope='Incomplete diagnostic; not a protocol performance result'),indent=2))
                break
        protocol_candidate=False
        if complete:
            report=summarize(root)
            (root/'result.json').write_text(json.dumps(report,indent=2))
            # Deliberately stop for model-level follow-up if the diagnostic
            # suggests a gain; never force a protocol globally or in long runs.
            protocol_candidate=large_message_candidate(report)
        exchange_candidates=[]
        if a.exchange_screen:
            from benchmarks.benchmark_bf16_exchange import audit_report
            exchange=a.output/'exchange';exchange.mkdir()
            for label,proto in (('auto','auto'),('simple','Simple')):
                idle('exchange_'+label,communication=True);status('running_exchange_screen',case=label)
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                for name in ('NCCL_PROTO','NCCL_ALGO','NCCL_DEBUG_FILE','NCCL_TOPO_DUMP_FILE','NCCL_GRAPH_DUMP_FILE'):
                    env.pop(name,None)
                if proto=='Simple':env['NCCL_PROTO']='Simple'
                command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                    'benchmarks/benchmark_bf16_exchange.py','--expected-proto',proto,'--output',str(exchange/label)]
                code,timed_out=bounded_run(command,exchange/(label+'.log'),env,timeout=600)
                if code or timed_out:
                    (exchange/(label+'_failure.json')).write_text(json.dumps(dict(returncode=code,timed_out=timed_out,
                        scope='Optional operator screen failed; no performance success or model installation'),indent=2))
                    continue
                measured=json.loads((exchange/label/'result.json').read_text())
                evaluated=audit_report(measured,proto)
                (exchange/(label+'_audit.json')).write_text(json.dumps(evaluated,indent=2))
                if any(evaluated['triton_over_two_percent_in_both_orders'][s] for s in ('decoder','head')):
                    exchange_candidates.append(label)
        if protocol_candidate or exchange_candidates:
            status('communication_candidate_requires_model_confirmation',protocol_candidate=protocol_candidate,
                   exchange_candidates=exchange_candidates);return
        idle('quality_followup')
        status('running_quality_followup',protocol_diagnostic_complete=complete)
        with (a.output/'quality.log').open('x') as stream:
            subprocess.run([sys.executable,'-m','scripts.queue_fsdp_final_quality',
                '--resume-pid',str(a.resume_pid),'--resume-queue',str(a.resume_queue),
                '--performance',str(a.performance),'--output',str(a.output/'quality')],
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        status('finished')
    except Exception as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
