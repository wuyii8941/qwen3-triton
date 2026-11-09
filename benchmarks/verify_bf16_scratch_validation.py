"""Single-GPU validator correctness and allocated-delta gate, not latency."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess

import torch

from benchmarks.bf16_scratch_validation import validate_upcast_flag
from scripts.screen_fsdp_1p7b import park_waiting_queue


def verify():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    free, util = [int(v.strip()) for v in rows[0].split(',')]
    if free < 16384 or util >= 10 or torch.cuda.device_count() != 1:
        raise ValueError('Expose idle physical GPU0 only, with16GiB free')
    props = torch.cuda.get_device_properties()
    torch.cuda.set_per_process_memory_fraction((8 * 1024**3) / props.total_memory)
    stream = torch.cuda.Stream()
    records = []
    def check(x, expected, label):
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.stream(stream):
            flag = validate_upcast_flag(x)
        stream.synchronize()
        delta = torch.cuda.max_memory_allocated() - baseline
        actual = flag.item()
        if actual != expected or delta > 65536:
            raise AssertionError(f'{label}: value={actual}, expected={expected}, delta={delta}')
        records.append(dict(case=label, count=x.numel(), expected=expected, actual=actual,
                            peak_extra_allocated_bytes=delta, nondefault_stream=True))
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    finite = bits[torch.isfinite(bits)].float().cuda()
    check(finite, 1, 'all_finite_bf16_patterns')
    del finite
    for count in (1, 17, 4095, 4096, 4097, 8193):
        x = torch.ones(count, device='cuda')
        check(x, 1, f'tail_{count}')
        for value, label in ((0.1, 'inexact'), (float('inf'), 'inf'), (float('nan'), 'nan')):
            x[-1] = value
            check(x, 0, f'tail_{count}_{label}')
        del x
    # Actual local8B decoder and LM-head element counts, without model loading.
    for name, count in (('decoder', 192946432), ('head', 622329856)):
        x = torch.ones(count, device='cuda')
        check(x, 1, f'real_shape_{name}_valid')
        x[-1] = 0.1
        check(x, 0, f'real_shape_{name}_invalid_tail')
        del x
    torch.cuda.synchronize()
    return dict(passed=True, records=records, gpu=props.name, torch_version=torch.__version__,
                allocator_cap_gib=8, latency_measured=False, fsdp_integrated=False,
                scope='Read-only validator correctness and Torch peak allocated delta excluding preexisting input; not reserved/NVML memory or full training benefit')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--park-pid', type=int, required=True)
    parser.add_argument('--park-status', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    def hashes():
        return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            'benchmarks/verify_bf16_scratch_validation.py', 'benchmarks/bf16_scratch_validation.py')}
    before = hashes()
    with park_waiting_queue(args.park_pid, args.park_status):
        result = verify()
        if hashes() != before:
            raise ValueError('Validator source changed')
        result['source_sha256'] = before
        with args.output.open('x') as file:
            json.dump(result, file, indent=2)
        print(json.dumps(dict(passed=True, cases=len(result['records']),
                             max_extra_allocated_bytes=max(r['peak_extra_allocated_bytes'] for r in result['records']))))


if __name__ == '__main__':
    main()
