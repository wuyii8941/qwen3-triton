"""Gated, fresh-model short training timing; no checkpoint/convergence claims."""
import gc
import argparse
import json
import statistics
import time
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.check_tied_embedding import patch
from benchmarks.research_candidates import make_optimizer
from benchmarks.screen_research_candidates import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-stage', choices=['baseline', 'candidate', 'multi', 'both', 'all', 'all_both','all_vec','all_vec_both','consume','all_vec_async','all_vec_async_both'])
    parser.add_argument('--output')
    parser.add_argument('--seq', type=int, default=512)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--phase-memory',action='store_true')
    parser.add_argument('--gate', default='artifacts/tied_embedding_deterministic_v1/result.json')
    args = parser.parse_args()
    gate = json.loads(Path(args.gate).read_text())
    if not gate['passed']:
        raise RuntimeError('Deterministic full-model gradient gate did not pass')
    if args.single_stage in ('candidate', 'both', 'all_both','all_vec_both','consume','all_vec_async_both') and (gate['seq'] != args.seq or gate.get('batch',1) != args.batch):
        raise RuntimeError('Embedding correctness gate does not cover requested shape')
    if args.steps < 8 or args.accum < 1:
        raise ValueError('Need at least 8 steps and positive accumulation')
    root = Path('artifacts/tied_embedding_step_v2')
    torch.set_num_threads(4)
    report = dict(scope='BF16 Qwen3-1.7B compiled forward/backward + FlashOptim; fixed validation batch, '
                        'fresh process and pretrained weights each stage, no checkpoint or convergence test.',
                  deterministic=False, torch=torch.__version__, rounds=[])
    if args.single_stage is None:
        root.mkdir(exist_ok=False)
        for round_id, stages in enumerate((('baseline', 'candidate'), ('candidate', 'baseline'))):
            for stage in stages:
                output = root / f'r{round_id}_{stage}.json'
                subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_tied_embedding_step',
                                '--single-stage', stage, '--output', str(output)], check=True)
                row = json.loads(output.read_text())
                row['round'] = round_id
                report['rounds'].append(row)
                (root/'result.json').write_text(json.dumps(report, indent=2))
        report['complete'] = True
        (root/'result.json').write_text(json.dumps(report, indent=2))
        return
    data = np.memmap('artifacts/convergence_corpus_v1/validation.bin', dtype='<u4', mode='r')
    x = torch.from_numpy(data[:args.seq*args.batch].astype(np.int64)).reshape(args.batch,args.seq).cuda()
    for round_id, stages in enumerate(((args.single_stage,),)):
        for stage in stages:
            torch.manual_seed(20261105)
            model = load_model()
            undo = patch(model,consume_proxy=stage=='consume') if stage in ('candidate','both','all_both','all_vec_both','consume','all_vec_async_both') else lambda: None
            if stage in ('multi','both','all','all_both','all_vec','all_vec_both','all_vec_async','all_vec_async_both'):
                from training.multi_tensor_flashoptim import MultiTensorFlashAdamW
                batching=dict(max_numel=2**31-1,persistent=True,vectorized=stage.startswith('all_vec'),async_metadata='async' in stage) if stage.startswith('all') else {}
                optimizer = MultiTensorFlashAdamW(model.parameters(), **batching, lr=1e-5, betas=(.9,.999),
                    eps=1e-8, weight_decay=.01, master_weight_bits=None, decouple_lr=False)
            else:
                optimizer = make_optimizer(model, 'flashoptim', 1e-5)
            def loss_fn(batch):
                return model(batch, labels=batch, use_cache=False).loss
            fn = torch.compile(loss_fn, fullgraph=True, dynamic=False)
            times, peaks, losses = [], [], []
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for i in range(args.steps):
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    for _ in range(args.accum):
                        loss = fn(x) / args.accum
                        loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)*1000)
                    peaks.append(torch.cuda.max_memory_allocated())
                    losses.append(float(loss.detach())*args.accum)
            row = dict(round=round_id, stage=stage, median_step_ms=statistics.median(times[4:]),
                       first_step_ms=times[0], all_step_ms=times,
                       peak_allocated_bytes=max(peaks[4:]), losses=losses,
                       seq=args.seq, batch=args.batch, accumulation=args.accum,
                       torch=torch.__version__, gpu=torch.cuda.get_device_name())
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        for _ in range(args.accum): (fn(x)/args.accum).backward()
                        optimizer.step()
                    torch.cuda.synchronize()
                row['profile_steps'] = 2
                row['events'] = [dict(name=e.key,count=e.count,self_device_us=e.self_device_time_total,
                                      self_cpu_us=e.self_cpu_time_total) for e in prof.key_averages()]
            if args.phase_memory:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                memory=dict(start_allocated=torch.cuda.memory_allocated())
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    loss=fn(x);torch.cuda.synchronize()
                    memory.update(forward_allocated=torch.cuda.memory_allocated(),forward_peak=torch.cuda.max_memory_allocated())
                    torch.cuda.reset_peak_memory_stats()
                    loss.backward();torch.cuda.synchronize()
                    memory.update(backward_allocated=torch.cuda.memory_allocated(),backward_peak=torch.cuda.max_memory_allocated())
                    torch.cuda.reset_peak_memory_stats()
                    optimizer.step();torch.cuda.synchronize()
                    memory.update(step_allocated=torch.cuda.memory_allocated(),step_peak=torch.cuda.max_memory_allocated())
                row['phase_memory_bytes']=memory
            report['rounds'].append(row)
            Path(args.output).write_text(json.dumps(row, indent=2))
            print('DONE', round_id, stage, row['median_step_ms'], row['peak_allocated_bytes'], flush=True)
            undo()
            del model, optimizer, fn, loss_fn, loss
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
