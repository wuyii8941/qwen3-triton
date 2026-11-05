"""Real Qwen3-1.7B ablation: static cache/graph, packed QKV, MLP, direct K.

FP16 is the acceptance target; BF16 still has a known >2% full-model drift in
the earlier baseline. No new candidate is silently promoted to the default.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
import torch
import transformers
import triton
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.attention import sdpa_kernel, SDPBackend
from model.hf_inference import qwen3_triton_inference, pack_qkv_weights, quantize_decode_mlp
from model.static_decode import StaticDecodeRunner
from ops.triton.weight_only_gemv import quantize_groupwise
from benchmarks.benchmark_hf_attention_e2e import check, bootstrap_speedup


VARIANTS = {
    'hf': {},
    'best': {},
    'packed': dict(packed_qkv=True),
    'prefill_mlp': dict(prefill_mlp=True),
    'packed_prefill': dict(packed_qkv=True, prefill_mlp=True),
    'static': dict(static_decode=True),
    'graph': dict(static_decode=True),
    'graph_packed': dict(static_decode=True, packed_qkv=True),
    'graph_direct': dict(static_decode=True, direct_k_cache=True),
    'graph_both': dict(static_decode=True, packed_qkv=True, direct_k_cache=True),
    'graph_all': dict(static_decode=True, packed_qkv=True, direct_k_cache=True, prefill_mlp=True),
}


def refinement_variants(tuned=False,dispatch=False,lm_head=False):
    base=dict(static_decode=True, packed_qkv=True, direct_k_cache=True, prefill_mlp=True)
    if lm_head:
        exact=dict(base,direct_v_cache=True,fuse_residual=True)
        selective=dict(exact,decode_int8_group=16,decode_int8_impl='grouped',decode_int8_max_batch=1)
        return {'hf':{},'best':{},'graph_ref':exact,
            'graph_head_contiguous':dict(exact,lm_head_impl='contiguous'),
            'graph_head_fp16':dict(exact,lm_head_impl='triton'),
            'graph_head64':dict(exact,lm_head_impl='int8',lm_head_group=64),
            'graph_head32':dict(exact,lm_head_impl='int8',lm_head_group=32),
            'graph_selective':selective,
            'graph_combined':dict(selective,lm_head_impl='int8',lm_head_group=64)}
    if dispatch:
        exact=dict(base,direct_v_cache=True,fuse_residual=True)
        return {'hf':{},'best':{},'graph_ref':base,'graph_exact':exact,
            'graph_selective':dict(exact,decode_int8_group=16,decode_int8_impl='grouped',decode_int8_max_batch=1)}
    if tuned:
        return {'hf':{},'best':{},'graph_ref':base,
            'graph_vcache':dict(base,direct_v_cache=True),
            'graph_exact':dict(base,direct_v_cache=True,fuse_residual=True),
            'graph_w8g32':dict(base,decode_int8_group=32,decode_int8_impl='grouped'),
            'graph_w8g16':dict(base,decode_int8_group=16,decode_int8_impl='grouped'),
            'graph_w8_all':dict(base,decode_int8_group=32,decode_int8_impl='grouped',direct_v_cache=True,fuse_residual=True)}
    return {'hf':{},'best':{},'graph_ref':base,
        'graph_native_mlp':dict(base,decode_mlp=False),
        'graph_addnorm':dict(base,fuse_residual=True),
        'graph_vcache':dict(base,direct_v_cache=True),
        **{f'graph_w8g{g}':dict(base,decode_int8_group=g) for g in (64,32,16)},
        'graph_w8_all':dict(base,decode_int8_group=64,direct_v_cache=True,fuse_residual=True)}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--tag', default='')
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--suite', choices=['original','refine','refine_v2','dispatch','lm_head'], default='original')
    parser.add_argument('--cases', default='1:128,2:128,1:512')
    args = parser.parse_args()
    variants=VARIANTS if args.suite=='original' else refinement_variants(args.suite=='refine_v2',args.suite=='dispatch',args.suite=='lm_head')
    baseline='best' if args.suite=='original' else 'graph_ref'
    if args.repeats < 8 or args.steps < 4:
        raise ValueError('Use at least 8 rounds and 4 decode steps')
    root = Path(__file__).resolve().parents[1]
    path = Path('/data1/tzh/models/Qwen/Qwen3-1.7B')
    target = root/'artifacts'/f'static_decode_e2e{args.tag}.json'
    if target.exists():
        raise FileExistsError(f'Preserve existing results; choose another --tag: {target}')
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').cuda().eval()
    torch.cuda.synchronize()
    start = time.perf_counter()
    packed = pack_qkv_weights(model)
    torch.cuda.synchronize()
    packing_ms = (time.perf_counter()-start)*1000
    packing_bytes = sum(t.numel()*t.element_size() for pair in packed.values() for t in pair if t is not None)
    quantized={}
    quant_setup={}
    for group in sorted({v['decode_int8_group'] for v in variants.values() if v.get('decode_int8_group')}):
        torch.cuda.synchronize()
        start=time.perf_counter()
        quantized[group]=quantize_decode_mlp(model,group)
        torch.cuda.synchronize()
        quant_setup[group]=dict(ms=(time.perf_counter()-start)*1000,
            extra_snapshot_bytes=sum(w.nbytes for pair in quantized[group].values() for w in pair))
    head_snapshots={}
    head_setup={}
    for group in sorted({v.get('lm_head_group',64) for v in variants.values() if v.get('lm_head_impl')=='int8'}):
        torch.cuda.synchronize();start=time.perf_counter()
        head_snapshots[group]=quantize_groupwise(model.lm_head.weight,group)
        torch.cuda.synchronize()
        head_setup[group]=dict(ms=(time.perf_counter()-start)*1000,extra_snapshot_bytes=head_snapshots[group].nbytes)
    def ctx(name):
        flags=dict(decode_mlp=True,fuse_qk=True,compact_gqa=True)
        flags.update(variants[name])
        group=flags.get('decode_int8_group',0)
        return nullcontext() if name == 'hf' else qwen3_triton_inference(
            model,packed_weights=packed,quantized_mlp=quantized.get(group),
            quantized_lm_head=head_snapshots.get(flags.get('lm_head_group',64)),**flags)
    sources = ['model/hf_inference.py', 'model/hf_attention.py', 'model/static_decode.py',
        'ops/triton/hf_qk_norm_rope.py', 'ops/triton/gemm_tuning.py',
        'ops/triton/gate_up_swiglu.py', 'ops/triton/decode_gate_up.py',
        'ops/triton/weight_only_gemv.py', 'ops/triton/add_rms_norm.py',
        'ops/triton/weight_only_tuning.py',
        'ops/triton/linear_decode.py',
        'benchmarks/benchmark_static_decode_e2e.py']
    report = dict(manifest=dict(
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
        transformers=transformers.__version__, dtype='float16', seed=args.seed,
        driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
            '--format=csv,noheader'], text=True).strip(), model=str(path),
        weights={p.name: sha(p) for p in sorted(path.glob('*.safetensors'))},
        source_sha256={p: sha(root/p) for p in sources}, config_sha256=sha(path/'config.json'),
        backend='forced native Flash; static decode uses pinned aten varlen Flash seqused_k',
        logits_to_keep=1, steps=args.steps, repeats=args.repeats,
        packing_ms=packing_ms, extra_packed_weight_bytes=packing_bytes,quantization_setup=quant_setup,lm_head_setup=head_setup,
        timing='randomized paired rounds; synchronized wall, CUDA events; warmup/setup excluded',
        pipeline='prefill + TTFT synchronization + fixed-length greedy decode; no EOS early exit',
        includes='LM head, argmax, cache update, input copy, graph replay, final synchronization',
        excludes='tokenizer, model loading, weight packing, JIT, graph capture; no torch.compile',
        memory='timed peak delta excludes preallocated cache/graph/packed weights, reported separately',
        reference=baseline,suite=args.suite,
        acceptance='existing relative L2 < 0.02 vs HF and best; random teacher-forced logits'),
        variants=variants, cases=[])
    for batch, seq in [tuple(map(int, c.split(':'))) for c in args.cases.split(',')]:
        ids = torch.randint(model.config.vocab_size, (batch, seq+8), device='cuda')
        prompt = ids[:, :seq].contiguous()
        capacity = triton.next_power_of_2(seq + max(args.steps, 8))
        case = dict(batch=batch, seq_len=seq, capacity=capacity, setup={}, parity={}, timings={})
        runners = {}
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for name, flags in variants.items():
                if not flags.get('static_decode'): continue
                with ctx(name):
                    torch.cuda.synchronize()
                    base = torch.cuda.memory_allocated()
                    start = time.perf_counter()
                    runner = StaticDecodeRunner(model, batch, capacity)
                    runner.prefill(prompt)
                    if name.startswith('graph'):
                        runner.capture(ids[:, seq:seq+1])
                    torch.cuda.synchronize()
                    case['setup'][name] = dict(ms=(time.perf_counter()-start)*1000,
                        cache_bytes=runner.cache.nbytes,
                        allocated_delta_bytes=torch.cuda.memory_allocated()-base)
                    runners[name] = runner
            # Teacher-forced comparisons isolate numerical changes from token divergence.
            references = {}
            for name in variants:
                with ctx(name):
                    if name in runners:
                        runner = runners[name]
                        out = runner.prefill(prompt)
                        logits = [out.clone()]
                        for pos in range(seq, seq+8):
                            out = runner.decode(ids[:, pos:pos+1], replay=name.startswith('graph'))
                            logits.append(out.clone())
                    else:
                        out = model(prompt, use_cache=True, logits_to_keep=1)
                        cache = out.past_key_values
                        logits = [out.logits.clone()]
                        for pos in range(seq, seq+8):
                            out = model(ids[:, pos:pos+1], past_key_values=cache,
                                        use_cache=True, logits_to_keep=1)
                            logits.append(out.logits.clone())
                        del cache
                    if name in ('hf', 'best',baseline): references[name] = logits
                    metrics = {ref: [check(a, b) for a, b in zip(logits, values)]
                               for ref, values in references.items()}
                    metrics['argmax_matches_hf'] = [bool(torch.equal(a.argmax(-1), b.argmax(-1)))
                                                   for a, b in zip(logits, references['hf'])]
                    metrics['accepted'] = all(p['passes_l2_gate'] for ref in references for p in metrics[ref])
                    case['parity'][name] = metrics
                    print('parity', batch, seq, name, metrics['accepted'],
                          max(p['relative_l2'] for p in metrics['hf']), flush=True)
                    del out, logits
            del references
            samples = {phase: {n: [] for n in variants} for phase in ('prefill', 'decode', 'pipeline')}
            gpu_samples = {phase: {n: [] for n in variants} for phase in ('prefill', 'decode')}
            peaks = {n: [] for n in variants}
            orders = []
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            for repeat in range(args.repeats+3):
                order = list(variants)
                rng.shuffle(order)
                if repeat >= 3: orders.append(order)
                for name in order:
                    with ctx(name):
                        torch.cuda.synchronize()
                        base = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        events[0].record()
                        if name in runners:
                            runner = runners[name]
                            logits = runner.prefill(prompt)
                        else:
                            out = model(prompt, use_cache=True, logits_to_keep=1)
                            logits, cache = out.logits, out.past_key_values
                        events[1].record()
                        torch.cuda.synchronize()
                        prefill_ms = (time.perf_counter()-start)*1000
                        decode_start = time.perf_counter()
                        events[2].record()
                        for step in range(args.steps):
                            token = logits[:, -1].argmax(-1, keepdim=True)
                            if name in runners:
                                logits = runner.decode(token, replay=name.startswith('graph'))
                            else:
                                out = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                                logits = out.logits
                        events[3].record()
                        torch.cuda.synchronize()
                        decode_ms = (time.perf_counter()-decode_start)*1000
                        pipeline_ms = (time.perf_counter()-start)*1000
                        if repeat >= 3:
                            for phase, value in [('prefill', prefill_ms), ('decode', decode_ms), ('pipeline', pipeline_ms)]:
                                samples[phase][name].append(value)
                            gpu_samples['prefill'][name].append(events[0].elapsed_time(events[1]))
                            gpu_samples['decode'][name].append(events[2].elapsed_time(events[3]))
                            peaks[name].append(torch.cuda.max_memory_allocated()-base)
                        if name not in runners: del out, cache
                        del logits, token
                if repeat >= 3 and (repeat-2) % 4 == 0:
                    print('rounds', batch, seq, repeat-2, '/', args.repeats, flush=True)
            case['order'] = orders
            for phase, values in samples.items():
                case['timings'][phase] = {
                    n: dict(median_ms=statistics.median(s), samples_ms=s,
                        **bootstrap_speedup(values[baseline], s)) for n, s in values.items()}
                for n in values:
                    if n.startswith('graph'):
                        graph_ref='graph' if args.suite=='original' else baseline
                        case['timings'][phase][n]['vs_graph'] = bootstrap_speedup(values[graph_ref], values[n])
            for phase, values in gpu_samples.items():
                for n, s in values.items():
                    case['timings'][phase][n]['cuda_event_median_ms'] = statistics.median(s)
            for n in variants:
                case['timings']['pipeline'][n]['peak_delta_bytes'] = max(peaks[n])
                case['timings']['decode'][n]['ms_per_step'] = statistics.median(samples['decode'][n])/args.steps
            if args.profile and batch == 1 and seq == 128:
                from torch.profiler import profile, ProfilerActivity
                case['profiler'] = {}
                profile_names=('best','static','graph','graph_packed','graph_direct','graph_both') if args.suite=='original' else ('graph_ref','graph_addnorm','graph_vcache','graph_w8g64','graph_w8_all')
                if args.suite=='refine_v2':profile_names=('graph_ref','graph_exact','graph_w8g32','graph_w8_all')
                if args.suite=='dispatch':profile_names=('graph_ref','graph_exact','graph_selective')
                if args.suite=='lm_head':profile_names=('graph_ref','graph_head_contiguous','graph_head64','graph_head_fp16','graph_combined')
                for name in profile_names:
                    with ctx(name):
                        if name in runners:
                            runner = runners[name]
                            runner.prefill(prompt)
                        else:
                            cache = model(prompt, use_cache=True, logits_to_keep=1).past_key_values
                        torch.cuda.synchronize()
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                            if name in runners:
                                runner.decode(ids[:, seq:seq+1], replay=name.startswith('graph'))
                            else:
                                model(ids[:, seq:seq+1], past_key_values=cache, use_cache=True, logits_to_keep=1)
                            torch.cuda.synchronize()
                        kernels = [e for e in prof.events() if str(e.device_type) == 'DeviceType.CUDA']
                        case['profiler'][name] = dict(cuda_kernel_count=len(kernels),
                            cuda_kernel_sum_us=sum(e.device_time_total for e in kernels),
                            kernels=dict(Counter(e.name for e in kernels)))
                        if name not in runners: del cache
            report['cases'].append(case)
            target.write_text(json.dumps(report, indent=2)+'\n')
            for phase in samples:
                print(batch, seq, phase, {n: round(t['median_ms'], 3)
                    for n, t in case['timings'][phase].items()}, flush=True)
        # Graph pools hold captured intermediates until all graph refs disappear.
        del runners, runner
        torch.cuda.empty_cache()
    # Natural-text smoke, separate from randomized timing and acceptance gate.
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    text = tokenizer.apply_chat_template([{'role':'user', 'content':'Explain GPU kernel fusion in one sentence.'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer(text, return_tensors='pt').input_ids.cuda()
    generated = {}
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        smoke_names=('hf','best','graph','graph_both','graph_all') if args.suite=='original' else tuple(variants)
        for name in smoke_names:
            with ctx(name):
                if name.startswith('graph'):
                    runner = StaticDecodeRunner(model, 1, prompt.shape[1]+32)
                    logits = runner.prefill(prompt)
                    runner.capture(logits[:, -1].argmax(-1, keepdim=True))
                else:
                    out = model(prompt, use_cache=True, logits_to_keep=1)
                    logits, cache = out.logits, out.past_key_values
                tokens = []
                for step in range(32):
                    token = logits[:, -1].argmax(-1, keepdim=True)
                    tokens.append(token.clone())
                    if name.startswith('graph'):
                        logits = runner.decode(token)
                    else:
                        out = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                        logits = out.logits
                generated[name] = torch.cat(tokens, dim=1)[0].tolist()
    report['generation_smoke'] = dict(tokens=generated,
        matches_hf={n: t == generated['hf'] for n, t in generated.items()},
        text={n: tokenizer.decode(t, skip_special_tokens=True) for n, t in generated.items()})
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(target, flush=True)


if __name__ == '__main__':
    main()
