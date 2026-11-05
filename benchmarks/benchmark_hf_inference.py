"""Real local Qwen3 checkpoint parity and matched prefill/decode timing."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import transformers
import triton
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.hf_inference import qwen3_triton_inference


def check(a, b):
    a, b = a.float(), b.float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise AssertionError("nonfinite output")
    relative_l2 = float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
    # Predeclared aggregate metric, not a claim of per-element/bitwise parity.
    assert relative_l2 < 0.02, relative_l2
    return dict(relative_l2=relative_l2, max_abs=float((a-b).abs().max()))


def measure(fn, prepare, repeats):
    samples, peaks = [], []
    for i in range(repeats+3):
        state = prepare()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = fn(state)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter()-start)*1000
        if i >= 3:
            samples.append(elapsed)
            peaks.append(torch.cuda.max_memory_allocated()-base)
        del result, state
    return dict(wall_median_ms=statistics.median(samples), wall_min_ms=min(samples),
                wall_max_ms=max(samples), samples_ms=samples, peak_delta_bytes=max(peaks))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data1/tzh/models/Qwen/Qwen3-1.7B')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--profile', action='store_true', help='Capture one untimed prefill per backend')
    args = parser.parse_args()
    torch.manual_seed(2026)
    path = Path(args.model)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                torch_dtype=getattr(torch, args.dtype), attn_implementation='sdpa').cuda().eval()
    def fingerprint(p):
        with p.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(manifest=dict(model=str(path), config_sha256=fingerprint(path/'config.json'),
             weights=[dict(name=p.name, size=p.stat().st_size, sha256=fingerprint(p)) for p in sorted(path.glob('*.safetensors'))],
             adapter_sha256=fingerprint(Path(__file__).resolve().parents[1]/'model/hf_inference.py'),
             driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
             torch=torch.__version__, transformers=transformers.__version__, triton=triton.__version__,
             cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), dtype=args.dtype,
             seed=2026, attention='forced FLASH_ATTENTION', logits_to_keep=1,
             scope='full pretrained model; supplied fixed tokens, no tokenizer/sampling timing',
             parity_threshold='relative L2 < 0.02; report max abs and argmax separately'), cases=[])
    for batch, seq in ((1, 128), (2, 128)):
        ids = torch.randint(0, model.config.vocab_size, (batch, seq+8), device='cuda')
        prompt = ids[:, :seq].contiguous()
        next_token = ids[:, seq:seq+1].contiguous()
        references, reference_states = None, None
        case = dict(batch=batch, prompt_len=seq, decode_steps=8, results={})
        for name in ('hf', 'triton'):
            context = nullcontext() if name == 'hf' else qwen3_triton_inference(model)
            with context, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = model(prompt, use_cache=True, logits_to_keep=1, output_hidden_states=True)
                states = [h.detach().clone() for h in out.hidden_states]
                logits = [out.logits.detach().clone()]
                cache = out.past_key_values
                for step in range(8):
                    out = model(ids[:, seq+step:seq+step+1], past_key_values=cache,
                                use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                    logits.append(out.logits.detach().clone())
                # Incremental decode and full causal forward must agree on the
                # same tokens/positions. Checks cache and position semantics.
                full = model(ids, use_cache=False, logits_to_keep=1).logits
                cache_parity = check(logits[-1], full)
                if references is None:
                    references, reference_states = logits, states
                parity = [check(a, b) for a, b in zip(logits, references)]
                layer_parity = [check(a, b) for a, b in zip(states, reference_states)]
                argmax = [float((a.argmax(-1)==b.argmax(-1)).float().mean()) for a,b in zip(logits,references)]
                del states, logits, out, cache, full

                def prefill(_):
                    return model(prompt, use_cache=True, logits_to_keep=1)

                def prepare():
                    return prefill(None).past_key_values

                def decode(cache):
                    return model(next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)

                result = dict(prefill=measure(prefill, lambda: None, args.repeats),
                              decode=measure(decode, prepare, args.repeats),
                              logits_parity=parity, hidden_states_parity=layer_parity,
                              full_vs_incremental=cache_parity, argmax_agreement=argmax)
                if args.profile and batch == 1:
                    from torch.profiler import profile, ProfilerActivity
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        prefill(None)
                        torch.cuda.synchronize()
                    result['profile_events'] = [dict(name=e.key, count=e.count,
                            device_time_us=e.device_time_total) for e in prof.key_averages()]
                case['results'][name] = result
                print(batch, seq, name, result['prefill']['wall_median_ms'],
                      result['decode']['wall_median_ms'], max(p['relative_l2'] for p in parity), flush=True)
        report['cases'].append(case)
    target = Path(__file__).resolve().parents[1]/f'artifacts/hf_inference_{args.dtype}.json'
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    main()
