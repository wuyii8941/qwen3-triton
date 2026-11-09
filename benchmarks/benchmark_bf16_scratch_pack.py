"""Single-GPU packing-only ABC/CBA probe; not communication/training timing."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess
import time

import torch

from benchmarks.bf16_scratch_pack import pack_owned_scratch
from benchmarks.bf16_scratch_bounded import pack_owned_scratch_bounded
from scripts.screen_fsdp_1p7b import park_waiting_queue


MODES = ('separate_bf16', 'scratch_debug', 'scratch_bounded')


@torch.no_grad()
def run(output):
    if output.exists():
        raise FileExistsError(output)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    free, util = [int(v.strip()) for v in rows[0].split(',')]
    if torch.cuda.device_count() != 1 or free < 20480 or util >= 10:
        raise ValueError('Expose idle physical GPU0 only, with20GiB free')
    props = torch.cuda.get_device_properties()
    torch.cuda.set_per_process_memory_fraction((12 * 1024**3) / props.total_memory)
    names = ('benchmarks/benchmark_bf16_scratch_pack.py', 'benchmarks/bf16_scratch_pack.py',
             'benchmarks/bf16_scratch_bounded.py', 'benchmarks/bf16_scratch_validation.py',
             'benchmarks/fsdp_scratch_exchange_adapter.py', 'benchmarks/bf16_exchange_reduce.py')
    def hashes():
        return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
    before = hashes()
    report = dict(complete=False, records=[], source_sha256=before, gpu=props.name,
                  torch_version=torch.__version__, allocator_cap_gib=12,
                  input_contract='FP32 constant1.25, exactly BF16 representable; rich inputs tested separately',
                  input_restore_excluded_from_op=True, input_restore_time_recorded=True,
                  workspace_baseline='Excludes immutable reference input and mutable scratch; includes returned send/receive and validation temporaries',
                  scope='Isolated single-GPU packing only; no NCCL/FSDP/optimizer/end-to-end speedup claim')
    with output.open('x') as file:
        json.dump(report, file, indent=2)
    for size, count in (('decoder', 192946432), ('head', 622329856)):
        source = torch.full((count,), 1.25, device='cuda', dtype=torch.float32)
        scratch = torch.empty_like(source)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # Initialize events before measured paths.
        start.record(); end.record(); torch.cuda.synchronize()
        for repeat, order in ((0, MODES), (1, tuple(reversed(MODES)))):
            for mode in order:
                torch.cuda.empty_cache()
                samples = []
                for index in range(15):
                    t_restore = time.perf_counter()
                    scratch.copy_(source)
                    torch.cuda.synchronize()
                    restore = time.perf_counter() - t_restore
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    tick = time.perf_counter()
                    start.record()
                    if mode == 'separate_bf16':
                        send = scratch.to(torch.bfloat16)
                        receive = torch.empty_like(send)
                    elif mode == 'scratch_debug':
                        send, receive = pack_owned_scratch(scratch, allow_mutation=True, backend='triton')
                    else:
                        send, receive = pack_owned_scratch_bounded(scratch, allow_mutation=True)
                    end.record()
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - tick
                    peak = torch.cuda.max_memory_allocated() - base
                    gpu_ms = start.elapsed_time(end)
                    # Correctness scan is intentionally AFTER timing/peak capture.
                    if not torch.all(send == 1.25).item() or send.numel() != receive.numel():
                        raise AssertionError('Packing output differs')
                    samples.append(dict(warmup=index < 3, wall_seconds=wall, cuda_ms=gpu_ms,
                                        peak_extra_allocated_bytes=peak, restore_seconds=restore))
                    del send, receive
                measured = [row for row in samples if not row['warmup']]
                row = dict(size=size, count=count, repeat=repeat, mode=mode, samples=samples,
                           median_wall_seconds=statistics.median(v['wall_seconds'] for v in measured),
                           median_cuda_ms=statistics.median(v['cuda_ms'] for v in measured),
                           peak_extra_allocated_bytes=max(v['peak_extra_allocated_bytes'] for v in samples),
                           fixed_input_values_checked=True)
                report['records'].append(row)
                output.write_text(json.dumps(report, indent=2))
                print(json.dumps({k: row[k] for k in ('size', 'repeat', 'mode', 'median_wall_seconds',
                                                      'peak_extra_allocated_bytes')}), flush=True)
        del source, scratch, start, end
        torch.cuda.empty_cache()
    if hashes() != before:
        raise ValueError('Packing benchmark source changed')
    report['complete'] = True
    output.write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--park-pid', type=int, required=True)
    parser.add_argument('--park-status', type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    with park_waiting_queue(args.park_pid, args.park_status):
        run(args.output)


if __name__ == '__main__':
    main()
