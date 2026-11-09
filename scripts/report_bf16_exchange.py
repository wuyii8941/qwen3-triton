"""Re-audit completed operator screens; never claim model performance."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.benchmark_bf16_exchange import audit_report


def candidates(evaluated):
    """Keep both implementations: using Torch locally is not disqualifying."""
    return [dict(size=size, backend=backend, paired_ratios=ratios)
            for size, modes in evaluated['paired_throughput_ratios'].items()
            if size in ('decoder', 'head')
            for backend, ratios in modes.items()
            if len(ratios) == 2 and all(value > 1.02 for value in ratios)]


def summarize(root):
    records = {}
    selected = []
    for label, proto in (('auto', 'auto'), ('simple', 'Simple')):
        path = root / label / 'result.json'
        raw = path.read_bytes()
        report = json.loads(raw)
        evaluated = audit_report(report, proto)
        records[label] = dict(path=str(path.resolve()),
            sha256=hashlib.sha256(raw).hexdigest(), audit=evaluated)
        selected.extend(dict(protocol_request=proto, **row) for row in candidates(evaluated))
    return dict(complete=True, screens=records, model_gate_candidates=selected,
        model_integrated=False, model_speedup_proven=False,
        scope='Random finite exact-upcast BF16 inputs; two ordered operator measurements on shared GPUs. '
              'Not statistical significance, arbitrary FP32 quantization, FSDP gradient parity, '
              'full-step memory/performance or convergence. Protocol is requested, not trace-verified.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report['model_gate_candidates'], indent=2))


if __name__ == '__main__':
    main()
