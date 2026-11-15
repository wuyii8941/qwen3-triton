"""Actual CUDA contract for the vocabulary-local full36-layer training graph."""
from collections import Counter
import hashlib,json
from qwen3_training_v11_contract import audit_trace as old_audit


def audit_trace(path,variant,vocab_ce,plan,checkpoint_every=1):
    config=dict(keep_full_layers=36,checkpoint_every=checkpoint_every,skip_embedding_backward_gather=True,retain_lm_head=True)
    if not vocab_ce:return old_audit(path,variant,config,plan)
    data=json.loads(path.read_text())
    events=[e for e in data['traceEvents'] if e.get('cat')=='kernel']
    counts=Counter(e['name'] for e in events)
    observed={kind:sum(n for name,n in counts.items() if 'ncclDevKernel_'+kind in name)
              for kind in ['AllGather','ReduceScatter','AllReduce']}
    # Remove the old head weight AG/gradient RS; add hidden, label and upstream
    # AG, one FP32 hidden-gradient RS and two small CE-statistic all-reduces.
    expected_collectives=dict(AllGather=42,ReduceScatter=39,AllReduce=2)
    collectives=dict(observed=observed,expected=expected_collectives,passed=observed==expected_collectives)
    head=dict(_gemm=6,_cast=2,_partials=1,_local_rows=1,_pack=1,_finish=1,_means=1,_grad=1,_accum=4)
    result=dict(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),collectives=collectives,vocab_ce=True)
    if variant=='hf':
        discrepancies={k:dict(expected=v,observed=counts[k]) for k,v in head.items() if counts[k]!=v}
        flash={k:v for k,v in counts.items() if 'flash' in k.lower()}
        result.update(native_decoder=True,custom_Triton_head=True,native_Flash_kernels=flash,
                      discrepancies=discrepancies,passed=bool(flash) and not discrepancies and collectives['passed'])
        return result
    assert variant=='triton_optimized'
    cp=len(range(0,36,checkpoint_every)) if checkpoint_every else 0
    expected=dict(_gemm=690+5*cp,_gate_up=36+cp,_native_precision_forward=36+cp,_native_split_dq=36,
        _dq_split_reduce=36,_native_precision_dkdv=36,_native_order_delta=36,
        _attention_kv_head_reduce_kernel=36,_hf_norm_forward=73+2*cp,_hf_swiglu_backward=36,
        _reduce_rows=290,_lookup=1,_zero=1,_positions=1,_tables=1,_backward=74,
        _add=72+2*cp,_hf_norm_rope_forward=72+2*cp,_norm_rope_backward=72,
        **{k:v for k,v in head.items() if k!='_gemm'})
    discrepancies={k:dict(expected=v,observed=counts[k]) for k,v in expected.items() if counts[k]!=v}
    framework=('FillFunctor','Copy<','split_with_sizes_copy_out_contiguous_no_cast_kernel',
               'direct_copy_kernel_cuda','chunk_cat_cuda_kernel','CUDAFunctor_add','ncclDevKernel_')
    unknown={k:v for k,v in counts.items() if k not in expected and not any(t in k for t in framework)}
    forbidden={k:v for k,v in unknown.items() if any(t in k.lower() for t in ('cublas','cutlass','gemm','flash','xmma'))}
    table={('gemm',e['M'],e['N'],e['K'],*(x for s in e['strides'] for x in s)):e
           for e in plan['entries'] if e['kind']=='gemm'} if plan else {}
    geometry=Counter()
    def add(m,n,k,am,ak,bk,bn,count,fp32=False):
        entry=None if fp32 else table.get(('gemm',m,n,k,am,ak,bk,bn))
        if entry:assert entry['family']=='tiled' and entry['splits']==1
        tm,tn,tk,group,warps,stages=entry['config'] if entry else (64,64,32,8,4,3)
        geometry[(( ((m+tm-1)//tm)*((n+tn-1)//tn),1,1),(warps*32,1,1))]+=count
    for out,inp,instances,fused,fwd in [(4096,4096,72,False,2*(36+cp)),(1024,4096,72,False,2*(36+cp)),
                                       (12288,4096,72,True,0),(4096,12288,36,False,36+cp)]:
        if not fused:add(512,out,inp,inp,1,1,inp,fwd)
        add(512,inp,out,out,1,inp,1,instances)
        add(out,inp,512,1,out,inp,1,instances)
    add(2048,37984,4096,4096,1,1,4096,1)
    add(2048,4096,37984,37984,1,4096,1,1,fp32=True)
    add(37984,4096,512,1,37984,4096,1,4)
    assert sum(geometry.values())==expected['_gemm']
    observed_geometry=Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name']=='_gemm')
    gate=next((e for e in plan['entries'] if e['kind']=='gate_up'),None) if plan else None
    tm,tn,tk,group,warps,stages=gate['config'] if gate else (64,64,32,8,4,3)
    gate_expected=Counter({((((512+tm-1)//tm)*((12288+tn-1)//tn),1,1),(warps*32,1,1)):36+cp})
    gate_observed=Counter((tuple(e['args']['grid']),tuple(e['args']['block'])) for e in events if e['name']=='_gate_up')
    geometry_passed=geometry==observed_geometry and gate_expected==gate_observed
    result.update(native_decoder=False,model_kernel_counts_passed=not discrepancies and not unknown,
                  discrepancies=discrepancies,unclassified_kernels=unknown,forbidden_library_kernels=forbidden,
                  launch_geometry_passed=geometry_passed,
                  passed=not discrepancies and not unknown and geometry_passed and collectives['passed'])
    return result
