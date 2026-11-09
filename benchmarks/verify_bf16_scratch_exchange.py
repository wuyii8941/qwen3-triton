"""Independent four-card scratch transport gate; not an FSDP callback."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch
import torch.distributed as dist

from benchmarks.bf16_scratch_pack import pack_owned_scratch
from benchmarks.bf16_exchange_reduce import sum_received
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.screen_fsdp_1p7b import park_waiting_queue


def source(rank, count):
    return (((torch.arange(count, dtype=torch.int64) + rank * 13) % 257 - 128)
            .float() / 16).bfloat16()


def hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        'benchmarks/verify_bf16_scratch_exchange.py', 'benchmarks/bf16_scratch_pack.py',
        'benchmarks/bf16_exchange_reduce.py')}


def worker(output):
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    if world != 4:
        raise ValueError('Four-rank gate required')
    torch.cuda.set_device(local)
    props = torch.cuda.get_device_properties(local)
    torch.cuda.set_per_process_memory_fraction((2 * 1024**3) / props.total_memory, local)
    before = hashes()
    dist.init_process_group('nccl', timeout=timedelta(seconds=240))
    records = []
    try:
        stream = torch.cuda.Stream()
        for cols in (1, 17, 129, 1025, 65539, 1048583):
            cpu = source(rank, world * cols)
            expected = torch.stack([source(r, world * cols)[rank * cols:(rank + 1) * cols]
                                    for r in range(world)])
            with torch.cuda.stream(stream):
                scratch = cpu.float().cuda()
                send, receive = pack_owned_scratch(scratch, allow_mutation=True, backend='triton')
                dist.all_to_all_single(receive, send)
                received_cpu = receive.view(world, cols).view(torch.int16).cpu()
                outputs = {average: sum_received(receive, world, average=average, backend='triton').cpu()
                           for average in (False, True)}
            stream.synchronize()
            if not torch.equal(received_cpu, expected.view(torch.int16)):
                raise AssertionError(f'rank{rank}, cols{cols}: transport bits differ')
            for average, actual in outputs.items():
                oracle = (expected.float() * (1 / world if average else 1)).sum(0)
                if not torch.equal(actual, oracle):
                    raise AssertionError(f'rank{rank}, cols{cols}: reduction differs')
            records.append(dict(cols=cols, transport_bits_equal=True,
                                fp32_sum_equal=True, fp32_avg_equal=True, nondefault_stream=True))
            del scratch, send, receive, outputs, cpu, expected, received_cpu
        torch.cuda.synchronize()
        if hashes() != before:
            raise ValueError('Sources changed during gate')
        result = dict(passed=True, rank=rank, world=world, records=records, source_sha256=before,
                      gpu=props.name, torch_version=torch.__version__,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(local))
        with (output / f'rank{rank}.json').open('x') as file:
            json.dump(result, file, indent=2)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--park-pid', type=int)
    parser.add_argument('--park-status', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.output)
        return
    if args.park_pid is None or args.park_status is None:
        raise ValueError('Waiting-only supervisor identity required')
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    args.output.mkdir(parents=True, exist_ok=False)
    before = hashes()
    with park_waiting_queue(args.park_pid, args.park_status):
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True).splitlines()
        resources = [[int(v.strip()) for v in row.split(',')] for row in rows[:4]]
        if len(resources) != 4 or any(free < 8192 or util >= 10 for free, util in resources):
            raise RuntimeError('Four idle cards with at least8GiB free required')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE'):
            env.pop(key, None)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   '-m', 'benchmarks.verify_bf16_scratch_exchange', '--worker', '--output', str(args.output)]
        code = run_owned(command, args.output / 'workers.log', env, timeout=600)
        if code:
            raise RuntimeError(f'Worker gate failed: {code}')
        records = [json.loads((args.output / f'rank{r}.json').read_text()) for r in range(4)]
        if hashes() != before or any(not r['passed'] or r['source_sha256'] != before or
                                     r['rank'] != rank or len(r['records']) != 6
                                     for rank, r in enumerate(records)):
            raise ValueError('Incomplete or changed gate records')
        result = dict(passed=True, world=4, rank_records=records, launch_resources=resources,
                      source_sha256=before, allocator_cap_gib_per_rank=2,
                      timing_measured=False, fsdp_integrated=False,
                      log_sha256=hashlib.sha256((args.output / 'workers.log').read_bytes()).hexdigest(),
                      scope='Small four-card NCCL transport + FP32 reduction correctness; not FSDP gradients, real-layer workspace or performance')
        with (args.output / 'result.json').open('x') as file:
            json.dump(result, file, indent=2)
        print(json.dumps(dict(passed=True, world=4, cases_per_rank=6, fsdp_integrated=False)))


if __name__ == '__main__':
    main()
