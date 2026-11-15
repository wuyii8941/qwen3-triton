"""CUDA kernel and collective contracts for full36-layer strategy experiments."""
from collections import Counter
import hashlib
import json


def audit_trace(path, variant, config, plan=None):
    data = json.loads(path.read_text())
    events = [event for event in data['traceEvents'] if event.get('cat') == 'kernel']
    counts = Counter(event['name'] for event in events)
    gathered = sum(count for name, count in counts.items() if 'ncclDevKernel_AllGather' in name)
    reduced = sum(count for name, count in counts.items() if 'ncclDevKernel_ReduceScatter' in name)
    expected_gathers = 78 - config['keep_full_layers'] - int(config['skip_embedding_backward_gather']) - int(config['retain_lm_head'])
    collectives = dict(passed=gathered == expected_gathers and reduced == 39,
                       expected_all_gather=expected_gathers, observed_all_gather=gathered,
                       expected_reduce_scatter=39, observed_reduce_scatter=reduced)
    result = dict(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), collectives=collectives)
    if variant == 'hf':
        flash = {name: count for name, count in counts.items() if 'flash' in name.lower()}
        result.update(reference=True, native_Flash_kernels=flash,
                      passed=bool(flash) and collectives['passed'])
        return result
    assert variant == 'triton_optimized'
    every = config['checkpoint_every']
    recomputed = len(range(0, 36, every)) if every else 0
    expected = dict(_gemm=687+5*recomputed, _gate_up=36+recomputed,
        _native_precision_forward=36+recomputed, _native_split_dq=36,
        _dq_split_reduce=36, _native_precision_dkdv=36, _native_order_delta=36,
        _attention_kv_head_reduce_kernel=36, _hf_norm_forward=73+2*recomputed,
        _hf_swiglu_backward=36, _reduce_rows=290, _lookup=1, _zero=1,
        _positions=1, _tables=1, _partials=1, _rows=1, _mean=1,
        _backward=75, _add=72+2*recomputed,
        _hf_norm_rope_forward=72+2*recomputed, _norm_rope_backward=72)
    discrepancies = {name: dict(expected=value, observed=counts[name])
                     for name, value in expected.items() if counts[name] != value}
    forbidden = {name: count for name, count in counts.items() if name not in expected
                 and any(token in name.lower() for token in ('cublas','cutlass','sgemm','hgemm','gemm','flash','xmma'))}
    framework_tokens = ('FillFunctor','Copy<','split_with_sizes_copy_out_contiguous_no_cast_kernel',
                        'direct_copy_kernel_cuda','chunk_cat_cuda_kernel','CUDAFunctor_add','ncclDevKernel_')
    unknown = {name: count for name, count in counts.items() if name not in expected
               and not any(token in name for token in framework_tokens)}
    table = {('gemm',entry['M'],entry['N'],entry['K'],*(v for stride in entry['strides'] for v in stride)): entry
             for entry in plan['entries'] if entry['kind'] == 'gemm'} if plan else {}
    geometry = Counter()
    for output, input_, instances, fused, forward_count in (
            (4096,4096,72,False,2*(36+recomputed)),
            (1024,4096,72,False,2*(36+recomputed)),
            (12288,4096,72,True,0),
            (4096,12288,36,False,36+recomputed),
            (151936,4096,1,False,1)):
        cases = []
        if not fused:
            cases.append((512,output,input_,input_,1,1,input_,forward_count))
        cases.extend(((512,input_,output,output,1,input_,1,instances),
                      (output,input_,512,1,output,input_,1,instances)))
        for m,n,k,am,ak,bk,bn,count in cases:
            entry = table.get(('gemm',m,n,k,am,ak,bk,bn))
            if entry:
                assert entry['family'] == 'tiled' and entry['splits'] == 1
            tm,tn,tk,group,warps,stages = entry['config'] if entry else (64,64,32,8,4,3)
            grid = ((m+tm-1)//tm)*((n+tn-1)//tn)
            geometry[((grid,1,1),(warps*32,1,1))] += count
    assert sum(geometry.values()) == expected['_gemm']
    observed = Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name'] == '_gemm')
    gate = next((e for e in plan['entries'] if e['kind'] == 'gate_up'), None) if plan else None
    tm,tn,tk,group,warps,stages = gate['config'] if gate else (64,64,32,8,4,3)
    gate_expected = Counter({((((512+tm-1)//tm)*((12288+tn-1)//tn),1,1),(warps*32,1,1)):36+recomputed})
    gate_observed = Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name'] == '_gate_up')
    geometry_passed = geometry == observed and gate_expected == gate_observed
    result.update(reference=False, recomputed_layers=recomputed,
                  model_kernel_counts_passed=not discrepancies and not forbidden and not unknown,
                  discrepancies=discrepancies, forbidden_library_kernels=forbidden,
                  unclassified_kernels=unknown, launch_geometry_passed=geometry_passed,
                  passed=not discrepancies and not forbidden and not unknown and geometry_passed and collectives['passed'])
    return result
