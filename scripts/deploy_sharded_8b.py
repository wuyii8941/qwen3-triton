"""Download/verify pretrained weights, then launch four-rank real-data smoke."""
import json
import os
import subprocess
import sys
import time
import argparse
import fcntl
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    parser = argparse.ArgumentParser()
    local = Path('/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
    original = Path('/data1/tzh/models/Qwen/Qwen3-8B')
    parser.add_argument('--model',type=Path,default=local if local.exists() else original)
    parser.add_argument('--output',type=Path,default=root/'artifacts/fsdp_8b_deployment_v1')
    parser.add_argument('--seq',type=int,default=512)
    parser.add_argument('--steps',type=int,default=3)
    options = parser.parse_args()
    if options.seq < 1 or options.steps < 1: parser.error('Positive seq/steps required')
    output = options.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'verification.json').exists(): raise FileExistsError('Already verified; select a new output directory')
    status = output/'pipeline_status.json'
    def update(phase,**extra):
        status.write_text(json.dumps(dict(phase=phase,updated=time.time(),pid=os.getpid(),
            model=str(options.model),**extra),indent=2))
        print(phase,flush=True)
    try:
        index = options.model/'model.safetensors.index.json'
        complete = index.exists() and all((options.model/name).is_file()
            for name in set(json.loads(index.read_text())['weight_map'].values()))
        if not complete:
            if options.model.resolve() != original.resolve(): raise FileNotFoundError('Local pretrained weights incomplete')
            update('downloading')
            with (output/'download.log').open('a') as stream:
                subprocess.run([sys.executable,'scripts/download_qwen3_8b.py'],stdout=stream,
                    stderr=subprocess.STDOUT,check=True)
            assert json.loads((original/'download_complete.json').read_text())['verified']
        # Do not launch over newly occupied cards. This is a same-node job.
        update('waiting_for_gpus')
        while True:
            usage = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
                '--format=csv,noheader,nounits'],text=True)
            if len(usage.splitlines())>=4 and all(int(v)<2048 for v in usage.splitlines()[:4]): break
            time.sleep(30)
        update('training')
        with (output/'training.log').open('a') as stream:
            subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
                '--nproc_per_node=4','training/run_sharded_qwen.py','--model',
                str(options.model),'--data','artifacts/convergence_corpus_v1/train.bin',
                '--steps',str(options.steps),'--seq',str(options.seq),'--reference-check',
                '--export-hf',str(output/'hf_model'),'--output',str(output)],stdout=stream,
                stderr=subprocess.STDOUT,check=True)
        records = [json.loads((output/f'rank{i}.json').read_text()) for i in range(4)]
        if not all(r['completed'] and not r['random_init'] and r['full_parameter_training'] and
                   r['global_parameters']>=7_000_000_000 and r['initial_hf_parity'] is not None and
                   len(r['records'])==options.steps for r in records):
            raise RuntimeError('Missing real pretrained multi-rank training evidence')
        update('reloading_export')
        with (output/'reload.log').open('a') as stream:
            subprocess.run([sys.executable,'benchmarks/verify_exported_qwen.py','--model',str(output/'hf_model'),
                '--reference',str(output/'export_reference.pt'),'--output',str(output/'verification.json')],
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        if not json.loads((output/'verification.json').read_text())['passed']:
            raise RuntimeError('Exported model verification failed')
        update('verified',parameters=sum(r['local_parameters'] for r in records),
            max_peak_allocated_gib=max(s['peak_allocated_gib'] for r in records for s in r['records']))
    except Exception as error:
        update('failed',error=str(error)); raise


if __name__ == '__main__': main()
