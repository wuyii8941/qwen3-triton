"""Single-GPU correctness gate, with safe parking of a waiting FSDP queue."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess

import torch

from benchmarks.bf16_scratch_pack import pack_owned_scratch
from scripts.screen_fsdp_1p7b import park_waiting_queue


def verify():
    # Caller selects physical GPU0; never start when that card is active.
    rows = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    free, util = [int(v.strip()) for v in rows[0].split(',')]
    if free < 16384 or util >= 10:
        raise RuntimeError('Physical GPU0 is not idle with sufficient memory')
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose physical GPU0 only using CUDA_VISIBLE_DEVICES=0')
    device = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction((2 * 1024**3) / device.total_memory)
    torch.cuda.reset_peak_memory_stats()
    stream = torch.cuda.Stream()
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    finite = patterns[torch.isfinite(patterns)]
    cases = [('all_finite_bf16_patterns', finite)]
    for count in (1, 2, 3, 15, 16, 17, 127, 129, 1025, 4097, 1048583):
        index = torch.arange(count, dtype=torch.int64) % finite.numel()
        cases.append((f'tail_{count}', finite[index]))
    records = []
    for name, source in cases:
        expected = source.view(torch.int16)
        with torch.cuda.stream(stream):
            # Cast on CPU so expected subnormal bits do not depend on a GPU cast.
            scratch = source.float().to('cuda')
            address = scratch.data_ptr()
            send, receive = pack_owned_scratch(scratch, allow_mutation=True, backend='triton')
            if send.data_ptr() != address or receive.data_ptr() != address + source.numel() * 2:
                raise AssertionError('Send/receive storage geometry differs')
            receive.zero_()
            actual = send.view(torch.int16).cpu()
        stream.synchronize()
        if not torch.equal(actual, expected):
            raise AssertionError(f'{name}: packed bits differ')
        records.append(dict(case=name, count=source.numel(), bits_equal=True,
                            receive_write_preserves_send=True, nondefault_stream=True))
        del scratch, send, receive, actual
    torch.cuda.synchronize()
    return dict(passed=True, records=records, gpu=device.name,
                compute_capability=list(torch.cuda.get_device_capability()),
                torch_version=torch.__version__, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                allocator_cap_bytes=2 * 1024**3, timing_measured=False,
                scope='Single-GPU packing correctness only; no NCCL/FSDP integration or speed/memory benefit claim')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--park-pid', type=int, required=True)
    parser.add_argument('--park-status', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    paths = [Path(__file__), Path('benchmarks/bf16_scratch_pack.py')]
    def hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    before = hashes()
    with park_waiting_queue(args.park_pid, args.park_status):
        result = verify()
        if hashes() != before:
            raise ValueError('Prototype source changed during verification')
        result['source_sha256'] = before
        with args.output.open('x') as output:
            json.dump(result, output, indent=2)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
