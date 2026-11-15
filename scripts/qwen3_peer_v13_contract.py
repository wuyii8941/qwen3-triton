"""Actual CUDA contract for full36-layer Qwen3 with peer FP32 reduction.

Checks raw events directly, never fabricates NCCL events or rewrites traces.
Default arms use the original V12 contract. Peer arms must replace exactly36
large decoder RS calls with checked-copy/peer-reduce and 72 real GPU barriers.
"""
from collections import Counter
import hashlib,json
from qwen3_vocab_v12_contract import audit_trace as baseline_audit


def audit_trace(path,variant,vocab_ce,plan,*,peer=False):
    if not peer:
        return baseline_audit(path,variant,vocab_ce,plan,1,fused_dw=vocab_ce,dw_config=(64,128,32,8,8,3) if vocab_ce else None)
    events=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel']
    counts=Counter(e['name'] for e in events)
    observed={kind:sum(n for name,n in counts.items() if 'ncclDevKernel_'+kind in name) for kind in ['AllGather','ReduceScatter','AllReduce']}
    expected_comm=dict(AllGather=42 if vocab_ce else 40,ReduceScatter=3,AllReduce=2 if vocab_ce else 0)
    barriers=[e for e in events if 'c10d::symmetric_memory::barrier_kernel' in e['name']]
    peer_counts=counts['_checked_wire_copy']==counts['_peer_reduce']==36 and len(barriers)==72
    peer_geometry=all(tuple(e['args']['grid'])==(1,1,1) and tuple(e['args']['block'])==(32,1,1) for e in barriers)
    n=192946432;q=n//4
    for name,size in [('_checked_wire_copy',n),('_peer_reduce',q)]:
        geometry=Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name']==name)
        peer_geometry=peer_geometry and geometry==Counter({(((size+1023)//1024,1,1),(128,1,1)):36})
    report=dict(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),peer=True,
        collectives=dict(observed=observed,expected=expected_comm,passed=observed==expected_comm),
        checked_copy_peer_reduce_and_barrier_counts_passed=peer_counts,peer_launch_geometry_passed=peer_geometry,
        expected_peer_calls=36,observed_peer_reduce=counts['_peer_reduce'],observed_checked_copy=counts['_checked_wire_copy'],observed_GPU_barriers=len(barriers))
    if variant=='hf':
        flash={k:v for k,v in counts.items() if 'flash' in k.lower()}
        report.update(native_decoder=True,native_Flash_kernels=flash,
                      passed=bool(flash) and peer_counts and peer_geometry and observed==expected_comm)
        return report
    if variant!='triton_optimized' or not vocab_ce:raise ValueError('Expected complete Triton optimized vocabulary-head arm')
    cp=36
    expected=dict(_gemm=686+5*cp,_gate_up=36+cp,_native_precision_forward=36+cp,_native_split_dq=36,
        _dq_split_reduce=36,_native_precision_dkdv=36,_native_order_delta=36,
        _attention_kv_head_reduce_kernel=36,_hf_norm_forward=73+2*cp,_hf_swiglu_backward=36,
        _reduce_rows=290,_lookup=1,_zero=1,_positions=1,_tables=1,_backward=74,
        _add=72+2*cp,_hf_norm_rope_forward=72+2*cp,_norm_rope_backward=72,
        _cast=2,_partials=1,_local_rows=1,_pack=1,_finish=1,_means=1,_grad=1,_accum=0,_vocab_dw_mean=1,
        _checked_wire_copy=36,_peer_reduce=36)
    discrepancies={k:dict(expected=v,observed=counts[k]) for k,v in expected.items() if counts[k]!=v}
    framework=('FillFunctor','Copy<','split_with_sizes_copy_out_contiguous_no_cast_kernel',
        'direct_copy_kernel_cuda','chunk_cat_cuda_kernel','CUDAFunctor_add','ncclDevKernel_',
        'c10d::symmetric_memory::barrier_kernel')
    unknown={k:v for k,v in counts.items() if k not in expected and not any(t in k for t in framework)}
    table={('gemm',e['M'],e['N'],e['K'],*(x for s in e['strides'] for x in s)):e for e in plan['entries'] if e['kind']=='gemm'}
    geometry=Counter()
    def add(m,n,k,am,ak,bk,bn,count,fp32=False):
        entry=None if fp32 else table.get(('gemm',m,n,k,am,ak,bk,bn))
        if entry:assert entry['family']=='tiled' and entry['splits']==1
        tm,tn,tk,group,warps,stages=entry['config'] if entry else (64,64,32,8,4,3)
        geometry[((((m+tm-1)//tm)*((n+tn-1)//tn),1,1),(warps*32,1,1))]+=count
    for out,inp,instances,fused,fwd in [(4096,4096,72,False,2*(36+cp)),(1024,4096,72,False,2*(36+cp)),
                                       (12288,4096,72,True,0),(4096,12288,36,False,36+cp)]:
        if not fused:add(512,out,inp,inp,1,1,inp,fwd)
        add(512,inp,out,out,1,inp,1,instances);add(out,inp,512,1,out,inp,1,instances)
    add(2048,37984,4096,4096,1,1,4096,1);add(2048,4096,37984,37984,1,4096,1,1,fp32=True)
    assert sum(geometry.values())==expected['_gemm']
    observed_geometry=Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name']=='_gemm')
    gate=next((e for e in plan['entries'] if e['kind']=='gate_up'),None)
    tm,tn,tk,group,warps,stages=gate['config'] if gate else (64,64,32,8,4,3)
    expected_gate=Counter({((((512+tm-1)//tm)*((12288+tn-1)//tn),1,1),(warps*32,1,1)):36+cp})
    observed_gate=Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name']=='_gate_up')
    fused_geometry=Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name']=='_vocab_dw_mean')==Counter({((((37984+63)//64)*((4096+127)//128),1,1),(256,1,1)):1})
    model_geometry=geometry==observed_geometry and expected_gate==observed_gate and fused_geometry
    report.update(native_decoder=False,model_kernel_counts_passed=not discrepancies and not unknown,
        discrepancies=discrepancies,unclassified_kernels=unknown,model_launch_geometry_passed=model_geometry,
        passed=not discrepancies and not unknown and model_geometry and peer_counts and peer_geometry and observed==expected_comm)
    return report
