"""Classify raw phase-wide kernels, with no framework-compute exemption."""
import hashlib,json
from collections import Counter
from qwen3_peer_v13_contract import audit_trace as previous_geometry_audit

RUNTIME_KERNELS={'_runtime_fill','_runtime_typed_fill','_runtime_strided_copy','_runtime_stage',
    '_runtime_peer_barrier','_runtime_peer_collective','_runtime_binary','_runtime_reduce',
    '_runtime_unary','_runtime_compare','_runtime_chunk_cat_one','_adamw_prepare','_adamw_update'}
MODEL_STEP=dict(_gemm=866,_gate_up=72,_native_precision_forward=72,_native_split_dq=36,
    _dq_split_reduce=36,_native_precision_dkdv=36,_native_order_delta=36,
    _attention_kv_head_reduce_kernel=36,_hf_norm_forward=145,_hf_swiglu_backward=36,
    _reduce_rows=290,_lookup=1,_zero=1,_positions=1,_tables=1,_backward=74,
    _add=144,_hf_norm_rope_forward=144,_norm_rope_backward=72,_cast=2,
    _partials=1,_local_rows=1,_pack=1,_finish=1,_means=1,_grad=1,_vocab_dw_mean=1)
MODEL_VALIDATION_BATCH=dict(_gemm=181,_gate_up=36,_native_precision_forward=36,
    _hf_norm_forward=73,_lookup=1,_positions=1,_tables=1,_add=72,_hf_norm_rope_forward=72,
    _cast=1,_partials=1,_local_rows=1,_pack=1,_finish=1,_means=1)


def audit_trace(path,*,phase,plan=None):
    raw=path.read_bytes();events=[e for e in json.loads(raw)['traceEvents'] if e.get('cat')=='kernel']
    counts=Counter(e['name'] for e in events);unknown={k:v for k,v in counts.items() if k not in RUNTIME_KERNELS and k not in MODEL_STEP}
    if phase in ['initial_step','full_step']:
        expected=MODEL_STEP;differences={name:dict(expected=n,observed=counts[name]) for name,n in expected.items() if counts[name]!=n}
        geometry=previous_geometry_audit(path,'triton_optimized',True,plan,peer=True)['model_launch_geometry_passed']
        optimizer=counts['_adamw_prepare']==counts['_adamw_update']==2
        communication=counts['_runtime_stage']>0 and counts['_runtime_peer_collective']>0 and counts['_runtime_peer_barrier']>=2*counts['_runtime_stage']
    elif phase=='validation':
        expected={name:n*32 for name,n in MODEL_VALIDATION_BATCH.items()}
        differences={name:dict(expected=n,observed=counts[name]) for name,n in expected.items() if counts[name]!=n}
        differences.update({name:dict(expected=0,observed=counts[name]) for name in set(MODEL_STEP)-set(expected) if counts[name]})
        geometry=None;optimizer=not counts['_adamw_prepare'] and not counts['_adamw_update'];communication=counts['_runtime_stage']>0 and counts['_runtime_peer_barrier']>0
    elif phase=='setup':
        differences={name:dict(expected=0,observed=counts[name]) for name in MODEL_STEP if counts[name]}
        geometry=None;optimizer=not counts['_adamw_prepare'] and not counts['_adamw_update'];communication=not counts['_runtime_peer_collective']
    else:raise ValueError('Explicit phase required')
    return dict(phase=phase,trace_sha256=hashlib.sha256(raw).hexdigest(),kernel_count=len(events),kernels=dict(counts),
        unclassified_or_native_compute=unknown,model_count_discrepancies=differences,model_launch_geometry_passed=geometry,
        optimizer_coverage_passed=optimizer,communication_coverage_passed=communication,
        passed=bool(events) and not unknown and not differences and optimizer and communication and geometry is not False,
        scope='Raw kernel events, no trace rewriting. Host CPU/driver/DMA events remain; every actual GPU compute kernel must be classified Triton. Control arms audited separately.')
