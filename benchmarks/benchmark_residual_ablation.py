"""Matched real-checkpoint residual-fusion ablation; alternating timing order.

Run with the project root on PYTHONPATH. Cache preparation and adapter setup
are outside timing; decode always starts from the same prompt length.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import transformers
import triton
from transformers import AutoModelForCausalLM
from torch.nn.attention import sdpa_kernel, SDPBackend
from model.hf_inference import qwen3_triton_inference
from benchmark_hf_inference import check


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dtype', choices=['float16', 'bfloat16'], default='bfloat16')
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--decode-mlp', action='store_true', help='Also ablate small-M fused GEMV')
    p.add_argument('--model', default='/data1/tzh/models/Qwen/Qwen3-1.7B')
    args = p.parse_args()
    torch.manual_seed(2026)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=getattr(torch, args.dtype), attn_implementation='sdpa').cuda().eval()
    def context(name):
        return nullcontext() if name == 'hf' else qwen3_triton_inference(
            model, fuse_residual=name == 'triton_residual',decode_mlp=name == 'triton_decode_mlp')
    root = Path(__file__).resolve().parents[1]
    report = dict(manifest=dict(dtype=args.dtype, model=args.model,
        torch=torch.__version__, triton=triton.__version__, transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, seed=2026,
        attention='forced Flash SDPA', timing='synchronized wall-time; alternating variant order',
        source_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
            ['model/hf_inference.py', 'ops/triton/add_rms_norm.py',
             'benchmarks/benchmark_residual_ablation.py']}), cases=[])
    names = ['hf', 'triton', 'triton_residual']
    if args.decode_mlp:
        names.append('triton_decode_mlp')
    for batch, seq in [(1,128), (2,128), (1,512)]:
        ids = torch.randint(model.config.vocab_size, (batch,seq+4), device='cuda')
        prompt = ids[:,:seq].contiguous()
        case = dict(batch=batch, seq_len=seq, parity={}, timings={})
        refs = None
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for name in names:
                with context(name):
                    out = model(prompt, use_cache=True, logits_to_keep=1, output_hidden_states=True)
                    states = [h.clone() for h in out.hidden_states]
                    logits = [out.logits.clone()]
                    cache = out.past_key_values
                    for step in range(4):
                        out = model(ids[:,seq+step:seq+step+1], past_key_values=cache,
                                    use_cache=True, logits_to_keep=1)
                        cache = out.past_key_values
                        logits.append(out.logits.clone())
                    full = model(ids, use_cache=False, logits_to_keep=1).logits
                    if refs is None:
                        refs = states, logits
                    case['parity'][name] = dict(hidden=[check(a,b) for a,b in zip(states,refs[0])],
                        logits=[check(a,b) for a,b in zip(logits,refs[1])],
                        argmax=[bool((a.argmax(-1)==b.argmax(-1)).all()) for a,b in zip(logits,refs[1])],
                        incremental=check(logits[-1],full))
                    del out, states, logits, cache, full
            del refs
            for phase in ['prefill', 'decode']:
                samples = {name: [] for name in names}
                peaks = {name: [] for name in names}
                for rep in range(args.repeats+4):
                    for name in (names if rep % 2 == 0 else names[::-1]):
                        with context(name):
                            cache = (model(prompt, use_cache=True, logits_to_keep=1).past_key_values
                                     if phase == 'decode' else None)
                            tokens = ids[:,seq:seq+1] if phase == 'decode' else prompt
                            torch.cuda.synchronize()
                            base = torch.cuda.memory_allocated()
                            torch.cuda.reset_peak_memory_stats()
                            start = time.perf_counter()
                            out = model(tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
                            torch.cuda.synchronize()
                            ms = (time.perf_counter()-start)*1000
                            if rep >= 4:
                                samples[name].append(ms)
                                peaks[name].append(torch.cuda.max_memory_allocated()-base)
                            del out, cache
                case['timings'][phase] = {name: dict(median_ms=statistics.median(samples[name]),
                    samples_ms=samples[name], peak_delta_bytes=max(peaks[name])) for name in names}
                print(args.dtype,batch,seq,phase,
                      {name: round(statistics.median(samples[name]),3) for name in names},flush=True)
        report['cases'].append(case)
    prefix = 'fusion_ablation' if args.decode_mlp else 'residual_ablation'
    target = root/'artifacts'/f'{prefix}_{args.dtype}.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    main()
