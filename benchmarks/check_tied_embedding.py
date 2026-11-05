"""Experimental tied-weight gradient transport; never patches the default trainer."""
import json
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.screen_research_candidates import load_model
from training.tied_embedding import TiedEmbedding, patch_tied_embedding as patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='tied_embedding_model_v1')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seq', type=int, default=512)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--gate-only', action='store_true')
    args = parser.parse_args()
    root = Path('artifacts') / args.tag
    root.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261105)
    torch.use_deterministic_algorithms(args.deterministic)
    report = dict(scope='Real Qwen BF16 compiled forward/backward; no optimizer or checkpoints.',
                  deterministic=args.deterministic, checks=[])
    try:
        model = load_model()
        data = np.memmap('artifacts/convergence_corpus_v1/validation.bin', dtype='<u4', mode='r')
        x = torch.from_numpy(data[:args.seq*args.batch].astype(np.int64)).reshape(args.batch, args.seq).cuda()
        report.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(), seq=args.seq, batch=args.batch,
                      unique_tokens=int(x.unique().numel()))
        reference = None
        for stage in ('baseline', 'repeat', 'candidate'):
            if stage == 'candidate':
                undo = patch(model)
            def loss_fn(batch):
                return model(batch, labels=batch, use_cache=False).loss
            fn = torch.compile(loss_fn, fullgraph=True, dynamic=False)
            model.zero_grad(set_to_none=True)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                loss = fn(x)
                loss.backward()
            torch.cuda.synchronize()
            grads = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
            if reference is None:
                reference = (float(loss.detach()), grads)
            else:
                total_error = total_norm = 0.
                rows = []
                for name, g in grads.items():
                    a, b = g.float(), reference[1][name].float()
                    err, norm = float((a-b).square().sum()), float(b.square().sum())
                    total_error += err
                    total_norm += norm
                    rows.append(dict(name=name, relative_l2=(err/max(norm, 1e-30))**.5))
                    if name == 'model.embed_tokens.weight':
                        hit = x.unique().cpu()
                        report[stage+'_hit_row_relative_l2'] = float((a[hit]-b[hit]).norm()/b[hit].norm())
                global_error = (total_error/max(total_norm, 1e-30))**.5
                worst = max(r['relative_l2'] for r in rows)
                check = dict(stage=stage, global_relative_l2=global_error, max_parameter_relative_l2=worst,
                             loss_abs=abs(float(loss.detach())-reference[0]),
                             worst_parameters=sorted(rows, key=lambda r:r['relative_l2'], reverse=True)[:5])
                check['passed'] = global_error <= .02 and worst <= .05 and check['loss_abs'] <= .01
                report['checks'].append(check)
                del grads
            print('STAGE', stage, report['checks'][-1:] , flush=True)
        report['passed'] = all(c['passed'] for c in report['checks'])
        undo()
        # Gate performance work on both repeat-control and candidate correctness.
        if report['passed'] and not args.gate_only:
            timings = []
            for stage in ('baseline', 'candidate', 'candidate', 'baseline'):
                undo = patch(model) if stage == 'candidate' else lambda: None
                fn = torch.compile(lambda batch: model(batch, labels=batch, use_cache=False).loss,
                                   fullgraph=True, dynamic=False)
                samples, peaks = [], []
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    for i in range(15):
                        model.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        fn(x).backward()
                        torch.cuda.synchronize()
                        if i >= 3:
                            samples.append((time.perf_counter()-start)*1000)
                            peaks.append(torch.cuda.max_memory_allocated())
                timings.append(dict(stage=stage, median_ms=statistics.median(samples), samples=samples,
                                    peak_allocated_bytes=max(peaks)))
                undo()
            report['forward_backward_timings'] = timings
    except Exception as exc:
        report.update(passed=False, error=f'{type(exc).__name__}: {exc}')
    (root/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
