"""CPU review helpers for actual full8B runs; never substitutes toy evidence."""
import json,math,statistics
from collections import Counter
from pathlib import Path
from qwen3_triton.gradient_archive import sha
from qwen3_runtime_v14_contract import audit_trace,MODEL_STEP
from qwen3_peer_v13_contract import audit_trace as v13_trace
from runtime_trace_v14 import TraceSource,locate_trace


ARMS={'HF_native':('native','hf'),'V13':('v13','triton_optimized'),
      'V13_TritonAdamW':('adam','triton_optimized'),'full_runtime':('full','triton_optimized')}


def trace_review(folder,runtime,plan,*,validation):
    if runtime=='full':
        phases=[('setup','setup_trace.json'),('initial_step','initial_step_trace.json'),('full_step','full_step_trace.json')]
        if validation:phases.append(('validation','full_validation_trace.json'))
        return {phase:audit_trace(locate_trace(folder,name),phase=phase,plan=plan) for phase,name in phases}
    result={}
    # Controls intentionally contain native GPU runtime kernels. Check their
    # actual model, communication and optimizer identity without describing
    # those native operations as Triton coverage.
    for phase in ['initial_step','full_step']:
        path=locate_trace(folder,phase+'_trace.json');source=TraceSource(path)
        events=[e for e in json.loads(source.read_text())['traceEvents'] if e.get('cat')=='kernel']
        counts=Counter(e['name'] for e in events)
        collectives={kind:sum(n for k,n in counts.items() if 'ncclDevKernel_'+kind in k) for kind in ['AllGather','ReduceScatter']}
        if runtime=='native':
            model=bool([k for k in counts if 'flash' in k.lower()])
            comm=collectives==dict(AllGather=40,ReduceScatter=39)
            geometry=None;differences={}
        else:
            old=v13_trace(source,'triton_optimized',True,plan,peer=True)
            differences={k:dict(expected=v,observed=counts[k]) for k,v in MODEL_STEP.items() if counts[k]!=v}
            geometry=old['model_launch_geometry_passed']
            model=not differences and geometry
            comm=(old['checked_copy_peer_reduce_and_barrier_counts_passed'] and old['peer_launch_geometry_passed']
                  and collectives==dict(AllGather=42,ReduceScatter=3))
        native_adam={k:v for k,v in counts.items() if 'adam' in k.lower() and k not in ['_adamw_prepare','_adamw_update']}
        optimizer=(counts['_adamw_prepare']==counts['_adamw_update']==2 and not native_adam) if runtime=='adam' else (bool(native_adam) and not counts['_adamw_update'])
        result[phase]=dict(trace_sha256=sha(path),decoded_trace_sha256=source.decoded_sha256,passed=bool(events) and bool(model) and bool(comm) and optimizer,
            model_passed=model,model_count_discrepancies=differences,model_launch_geometry_passed=geometry,
            communication_passed=comm,native_collectives=collectives,optimizer_passed=optimizer,native_AdamW_kernels=native_adam,
            kernels=dict(counts),scope='Control arm permits native runtime; model/optimizer/communication identities checked explicitly.')
    return result


def numerical_review(g,*,strict,canonical=None,norms=None,reference_sha=None):
    assert g['complete'] and g['strict_native_control']==strict and len(g['rows'])==399
    assert len({r['name'] for r in g['rows']})==399
    if canonical is not None:assert {r['name'] for r in g['rows']}==canonical
    if reference_sha is not None:assert g['reference_result_sha256']==reference_sha
    for r in g['rows']:
        assert all(math.isfinite(r[k]) and r[k]>=0 for k in ['absolute_error_norm','reference_norm','relative_l2','threshold'])
        if norms is not None:assert r['reference_norm']==norms[r['name']]
        threshold=(.0001 if strict else .06)*max(r['reference_norm'],1e-8)+(1e-7 if strict else 1e-6)
        assert r['threshold']==threshold and r['passed']==(r['absolute_error_norm']<threshold)
        assert r['relative_l2']==r['absolute_error_norm']/max(r['reference_norm'],1e-8)
    error=sum(r['absolute_error_norm']**2 for r in g['rows']);norm=sum(r['reference_norm']**2 for r in g['rows'])
    assert g['global_relative_l2']==(error/max(norm,1e-16))**.5
    gradient=all(r['passed'] for r in g['rows'])
    assert math.isfinite(g['logits_relative_l2']) and math.isfinite(g['loss_abs_error'])
    forward=g['logits_relative_l2']<.025 and g['loss_abs_error']<.01 and (g['forward_bitwise_equal'] or not strict)
    assert g['passed']==gradient and g['forward_gate_passed']==forward and g['numerical_gate_passed']==(gradient and forward)
    return dict(passed=gradient and forward,all399=True,gradient_relative_l2=g['global_relative_l2'],
        forward_bitwise_equal=g['forward_bitwise_equal'],failed_gradients=[r['name'] for r in g['rows'] if not r['passed']])


def run_review(d,role,*,formal,reference_sha,canonical=None,norms=None):
    runtime,variant=ARMS[role['arm']]
    assert d['complete'] and d['model']=='Qwen/Qwen3-8B' and d['layers']==36 and d['world_size']==4 and d['official_parameter_count']==8190735360
    assert d['runtime']==runtime and d['variant']==variant
    assert d['math_policy']==dict(compute='BF16',master_gradient_moments='FP32',TF32=False,allow_bf16_reduced_precision_reduction=False)
    assert d['common_framework']==dict(checkpoint_every=1,keep_full_layers=36,skip_embedding_backward_gather=True,
        retain_lm_head=variant=='hf',vocab_parallel=variant!='hf',fused_vocab_dw=variant!='hf',optimizer_groups='decoder+head_in_all_arms',
        optimizer='TritonAdamW' if runtime in ['adam','full'] else 'native_fused_AdamW',batch_per_rank=1,sequence=512)
    p=d['protocol'];quality=p['quality_steps'];warm=p['warmup_steps'];timed=p['measured_steps'];total=quality+warm+timed+1
    assert p['real_profile_updates']==1 and p['total_optimizer_updates']==total
    if formal:assert (quality,warm,timed,total)==(16,3,8,28) and not d['screen_only']
    has_validation=bool(p['validation_points'])
    assert p['validation_points']==([0,quality] if has_validation else [])
    assert p['validation_blocks']==(512 if has_validation else 0)
    if formal:assert has_validation
    assert len(d['ranks'])==4
    numerics=[];memory=[];transport=[];samples=None
    for rank,row in enumerate(d['ranks']):
        assert row['rank']==rank and row['complete'] and row['updates']==total and row['parameter_count']==row['optimizer_state_count']==399
        assert row['all_master_grad_moment_finite_fp32']
        assert [r['step'] for r in row['rows']]==list(range(1,total+1))
        for r in row['rows']:
            assert r['measured']==(quality+warm<r['step']<=quality+warm+timed)
            assert r['profile_update']==(r['step']==total)
            assert math.isfinite(r['global_loss']) and math.isfinite(r['wall_seconds']) and r['wall_seconds']>0
        own=[r['wall_seconds'] for r in row['rows'] if r['measured']];assert len(own)==timed
        if samples is None:samples=own
        assert samples==own and row['rows']==d['ranks'][0]['rows']
        assert row['full_validation']==d['ranks'][0]['full_validation']
        assert [v['after_updates'] for v in row['full_validation']]==p['validation_points']
        assert all(v['blocks']==512 and v['tokens']==261632 and v['batch_per_rank']==4 and math.isfinite(v['loss']) for v in row['full_validation'])
        assert row['vocab_head_calls']==(None if variant=='hf' else dict(forward=total+64*has_validation,backward=total))
        assert len(row['gradient_comparisons'])==1
        numerics.append(numerical_review(row['gradient_comparisons'][0],strict=variant=='hf',
            canonical=None if canonical is None else canonical[rank],norms=None if norms is None else norms[rank],reference_sha=reference_sha))
        report=row['runtime'];assert report['kind']==runtime and report['CPU_bootstrap_and_metadata_backend']=='gloo'
        extra=0 if runtime=='native' else .5
        assert row['effective_training_peak']=={k:v+extra for k,v in row['measured_training_peak'].items()}
        ledger=report['phase_memory_ledger']
        expected_phases=['setup']+(['validation_0'] if has_validation else [])+['quality_updates']+(['validation_after_quality'] if has_validation else [])+['warmup_updates','timed_updates','profile_update','post_update_checks']
        assert [r['phase'] for r in ledger]==expected_phases
        for event in ledger:
            ex=0 if runtime in ['v13','adam'] and event['phase'].startswith('validation') else extra
            assert event['external_budget_bytes']==ex*2**30
            assert event['effective']=={k:v+ex for k,v in event['torch'].items()}
        assert row['all_phase_peak']=={k:max(r['torch'][k] for r in ledger) for k in ['allocated_gib','reserved_gib']}
        assert row['effective_all_phase_peak']=={k:max(r['effective'][k] for r in ledger) for k in ['allocated_gib','reserved_gib']}
        memory.append(all(math.isfinite(v) and 0<=v<=41.875 for field in ['effective_training_peak','effective_all_phase_peak'] for v in row[field].values()))
        comm=report['communication']
        if runtime=='full':
            assert report['strict_GPU_dispatch'] is not None
            assert comm['external_memory_budget_bytes']==512*2**20 and comm['observed_extra_process_bytes']<=512*2**20
            assert comm['torch_allocator_cap_GiB']==41.375 and comm['total_effective_cap_GiB']==41.875
            # Exact communication field/error checks are tied to the actual
            # runtime report schema, rather than inferred from test success.
            assert comm['bootstrap_backend']=='gloo' and comm['GPU_data_backend']=='Triton' and comm['GPU_barrier_backend']=='Triton system release/acquire'
            assert comm['native_GPU_collective_calls']==comm['native_CUDA_barrier_calls']==0
            assert set(comm['channels'])=={'gather','scatter','reduce','alltoall'}
            assert sum(c['mapped_buffer_and_signal_bytes'] for c in comm['channels'].values())<=512*2**20
            transport.append(comm['timeout_flag']==comm['lossless_violation_flag']==0)
            assert comm['FSDP_AG_calls']>0 and comm['FSDP_RS_calls']>0
        elif runtime in ['v13','adam']:
            assert comm['peer_calls']==comm['calls']==36*total and comm['native_calls']==2*total
            assert comm['checked_elements']==comm['peer_calls']*192946432
            assert comm['external_memory_budget_bytes']==512*2**20 and comm['observed_extra_process_bytes']<=512*2**20
            assert comm['lossless_violation_flag']==0
            assert len(comm['validation_cycles'])==2*has_validation
            transport.append(comm['lossless'] and comm['guard_positive_negative_selftest_passed'])
        else:assert comm==dict(mode='native');transport.append(True)
    gates=dict(numerics=all(r['passed'] for r in numerics),effective_memory=all(memory),lossless_transport=all(transport))
    return dict(gates=gates,numerics=numerics,memory=memory,transport=transport,samples_seconds=samples,median_seconds=statistics.median(samples))
