"""Screen SM86 GEMMs on official 8B weights and captured real activations.

This is operator screening, not a whole-step speed or model-quality result.
Configurations are resource-pruned, then measured; finalists receive a second
measurement in reverse order. Outputs are compared with strict BF16/FP32 cuBLAS.
Allocation and deterministic filling are outside these raw kernel timings.
"""
from pathlib import Path
import argparse, gc, hashlib, json, statistics, time
import torch
import torch.nn.functional as F
import triton
from safetensors import safe_open
from qwen3_triton.kernels.linear import _gemm
from qwen3_triton.kernels.mlp_precision import _gate_up
from qwen3_triton.kernels.gemm_experiments import split_matmul_into, _gate_up_accum
from qwen3_triton.gradient_archive import sha

ROOT = Path(__file__).resolve().parents[1]
BASE = (64, 64, 32, 8, 4, 3)
CONFIGS = [BASE,
    (64,128,32,8,4,3), (64,128,64,8,4,3), (64,128,64,8,8,3),
    (32,128,64,8,4,3), (128,64,32,8,4,3), (128,64,64,8,8,3),
    (128,128,32,8,8,3), (128,128,64,8,8,2), (32,64,64,8,4,4),
    (64,64,64,8,4,3), (64,64,128,8,4,2), (32,128,128,8,4,2),
    (32,256,64,8,8,2), (64,256,32,8,8,3), (64,128,64,8,4,2),
    (64,128,64,8,4,4), (64,128,64,1,4,3), (64,128,64,4,4,3),
    (128,128,32,4,4,3), (32,64,32,8,4,3), (64,64,32,8,4,4),
    (32,128,32,8,4,4), (64,64,64,1,4,3)]


def measure(fn, warm=3, repeats=12):
    for _ in range(warm):
        fn()
    stamps = []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record(); fn(); end.record(); stamps.append((start, end))
    torch.cuda.synchronize()
    values = [start.elapsed_time(end) for start, end in stamps]
    return dict(samples_ms=values, median_ms=statistics.median(values))


def error(actual, reference):
    a, b = actual.view(-1), reference.view(-1)
    numerator, denominator, maximum, different = 0., 0., 0., 0
    finite = True
    for offset in range(0, a.numel(), 2**20):
        x, y = a[offset:offset+2**20].float(), b[offset:offset+2**20].float()
        d = x - y
        finite &= bool(torch.isfinite(x).all())
        numerator += float(d.square().sum())
        denominator += float(y.square().sum())
        maximum = max(maximum, float(d.abs().max()))
        different += int((d != 0).sum())
    relative = (numerator / max(denominator, 1.e-30)) ** .5
    return dict(finite=finite, relative_l2=relative, max_abs=maximum,
                differing_elements=different, passed=finite and relative < .001)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/data1/tzh/models/Qwen/Qwen3-8B'))
    parser.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert not out.is_relative_to(Path('/home/tzh'))
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.cuda.set_device(0); torch.manual_seed(20261114)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    prop = torch.cuda.get_device_properties(0)
    assert (prop.major, prop.minor) == (8, 6), prop
    torch.cuda.set_per_process_memory_fraction(41.875 * 2**30 / prop.total_memory)
    proof = json.loads(args.capture.with_suffix('.json').read_text())
    assert proof['complete'] and proof['layers'] == 36 and proof['parameter_tensors'] == 399
    assert sha(args.capture) == proof['capture_sha256']
    capture = torch.load(args.capture, map_location='cpu', weights_only=True)['reference']
    index = json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    config = json.loads((args.model/'config.json').read_text())
    assert config['num_hidden_layers'] == 36 and config['hidden_size'] == 4096

    def weight(name):
        with safe_open(args.model/index[name], framework='pt', device='cpu') as file:
            return file.get_tensor(name).cuda()

    rows = []
    provenance = dict(model='Qwen/Qwen3-8B', layers=36,
        revision='b968826d9c46dd6066d109eabc6255188de91218',
        hardware=prop.name, capability=[prop.major, prop.minor],
        capture_sha256=proof['capture_sha256'], publication_baseline='13dbdf984201a70ca4bba5fb38c72606cd33fbeb',
        sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/'qwen3_triton/kernels/linear.py', ROOT/'qwen3_triton/kernels/mlp_precision.py',
            ROOT/'qwen3_triton/kernels/gemm_experiments.py']},
        numerical_screen_relative_l2_limit=.001,
        search='Resource-pruned configuration seeds; no Helion installation or claimed reproduction of Helion results',
        references=['https://pytorch.org/blog/fast-by-default-compile-time-heuristics-for-helion-kernels/',
                    'https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html',
                    'https://arxiv.org/abs/2301.03598'], scope=__doc__)

    def save(complete=False):
        (out/('result.json' if complete else 'progress.json')).write_text(
            json.dumps(dict(complete=complete, provenance=provenance, rows=rows), indent=2)+'\n')

    for kind in ('q_proj', 'k_proj', 'gate_proj', 'down_proj', 'lm_head', 'gate_up_fused'):
        mlp = kind in ('gate_proj', 'down_proj', 'gate_up_fused')
        if kind == 'lm_head':
            x = capture['lm_head']['input'].cuda().view(512, 4096)
            w = weight('lm_head.weight')
        else:
            source = 'mlp' if mlp else 'self_attn'
            x = capture[f'model.layers.0.{source}']['input'].cuda().view(512, 4096)
            w = weight(f'model.layers.0.{source}.{kind if kind != "gate_up_fused" else "gate_proj"}.weight')
        if kind in ('down_proj', 'gate_up_fused'):
            wg, wu = (weight('model.layers.0.mlp.'+name+'.weight') for name in ('gate_proj', 'up_proj'))
            g, u = F.linear(x, wg), F.linear(x, wu)
            y = F.silu(g) * u
            if kind == 'down_proj':
                x = y
                del wg, wu, g, u, y
        dy = torch.randn((512, w.shape[0]), device='cuda', dtype=torch.bfloat16)
        for direction in (('fused_forward',) if kind == 'gate_up_fused' else ('forward','dx','dw')):
            fused = direction == 'fused_forward'
            if not fused:
                a, b = {'forward':(x,w.t()), 'dx':(dy,w), 'dw':(dy.t(),x)}[direction]
                m, k = a.shape; n = b.shape[1]
                reference = torch.mm(a, b)
                c = torch.empty_like(reference)
            else:
                a = x; m, k = x.shape; n = wg.shape[0]
                references = (g, u, y)
                outputs = tuple(torch.empty_like(t) for t in references)
            records, functions, workspaces = [], {}, []
            shared_limit = prop.shared_memory_per_block_optin
            specs = [('v9', BASE, 1)] + [('tiled', cfg, 1) for cfg in CONFIGS if cfg != BASE]
            if fused:
                # CPU IR proves V9 already carries both Tensor Core accumulators;
                # keep one negative control instead of duplicating the search.
                specs += [('explicit_accum', BASE, 1)]
            if not fused and k >= 12288 and m == 512:
                specs += [('split_k', cfg, splits) for cfg in ((64,64,64,8,4,3),(32,128,64,8,4,3)) for splits in (2,4,8)]
            for family, cfg, splits in specs:
                tm, tn, tk, group, warps, stages = cfg
                # A rough feasibility filter; compilation supplies exact resources.
                shared_estimate = (tm + (2*tn if fused else tn)) * tk * 2 * stages
                if shared_estimate > shared_limit:
                    continue
                label = f'{family}_{"_".join(map(str,cfg))}_split{splits}'
                item = dict(label=label, family=family, config=cfg, splits=splits)
                partials = None
                if family == 'split_k':
                    partials = torch.empty((splits,m,n),device='cuda',dtype=torch.float32)
                    workspaces.append(partials)
                    def fn(cfg=cfg,splits=splits,partials=partials):
                        return split_matmul_into(a,b,c,partials,cfg,splits)
                elif fused:
                    def fn(cfg=cfg,family=family):
                        tm,tn,tk,group,warps,stages=cfg
                        impl = _gate_up_accum if family == 'explicit_accum' else _gate_up
                        return impl[(triton.cdiv(m,tm)*triton.cdiv(n,tn),)](
                            x,wg,wu,*outputs,m,n,k,tm,tn,tk,group,
                            num_warps=warps,num_stages=stages,enable_fp_fusion=False)
                else:
                    def fn(cfg=cfg):
                        tm,tn,tk,group,warps,stages=cfg
                        return _gemm[(triton.cdiv(m,tm)*triton.cdiv(n,tn),)](
                            a,b,c,m,n,k,*a.stride(),*b.stride(),tm,tn,tk,group,
                            num_warps=warps,num_stages=stages)
                try:
                    kernel = fn(); torch.cuda.synchronize()
                    checks = [error(actual,ref) for actual,ref in zip(outputs,references)] if fused else [error(c,reference)]
                    item.update(numerical=checks, numerical_passed=all(t['passed'] for t in checks),
                        registers=kernel.n_regs, spills=kernel.n_spills, shared_bytes=kernel.metadata.shared,
                        partial_workspace_bytes=0 if partials is None else partials.numel()*4)
                    if item['numerical_passed']:
                        item['screen']=measure(fn)
                        functions[label]=fn
                except triton.OutOfResources as exc:
                    item['resource_rejected']=str(exc)
                records.append(item)
            valid = [r for r in records if r.get('numerical_passed') and 'screen' in r]
            baseline = next(r for r in valid if r['family']=='v9')
            finalists = sorted(valid,key=lambda r:r['screen']['median_ms'])[:3]
            if baseline not in finalists:
                finalists.append(baseline)
            if not fused:
                for family in ('split_k',):
                    candidates=[r for r in valid if r['family']==family]
                    if candidates:
                        candidate=min(candidates,key=lambda r:r['screen']['median_ms'])
                        if candidate not in finalists:finalists.append(candidate)
            else:
                for family in ('tiled','explicit_accum'):
                    candidates=[r for r in valid if r['family']==family]
                    if candidates:
                        candidate=min(candidates,key=lambda r:r['screen']['median_ms'])
                        if candidate not in finalists:finalists.append(candidate)
            for order in (1,2):
                candidates=finalists if order==1 else list(reversed(finalists))
                for item in candidates:
                    item[f'confirmation_order{order}']=measure(functions[item['label']],warm=5,repeats=25)
                if not fused:
                    baseline[f'native_order{order}']=measure(lambda:torch.mm(a,b,out=c),warm=5,repeats=25)
            winner=min(finalists,key=lambda r:sum(r[f'confirmation_order{o}']['median_ms'] for o in (1,2)))
            row=dict(module=kind,direction=direction,M=m,N=n,K=k,
                     strides=None if fused else [list(a.stride()),list(b.stride())],
                     records=records,winner=winner['label'],baseline=baseline['label'])
            rows.append(row);save()
            print(json.dumps(dict(module=kind,direction=direction,winner=winner['label'],
                baseline_ms=[baseline[f'confirmation_order{o}']['median_ms'] for o in (1,2)],
                winner_ms=[winner[f'confirmation_order{o}']['median_ms'] for o in (1,2)],
                native_ms=None if fused else [baseline[f'native_order{o}']['median_ms'] for o in (1,2)])),flush=True)
            functions.clear();workspaces.clear()
            if not fused:del a,b,c,reference
            else:del outputs,references
            gc.collect();torch.cuda.empty_cache()
        del x,w,dy
        if kind=='gate_up_fused':del wg,wu,g,u,y
        gc.collect();torch.cuda.empty_cache()
    assert len(rows)==16
    save(complete=True)


if __name__=='__main__':
    main()
