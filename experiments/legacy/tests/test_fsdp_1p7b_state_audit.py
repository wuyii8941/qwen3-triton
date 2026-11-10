import copy
import hashlib
import json
import struct
import pytest
from scripts.audit_fsdp_1p7b_states import check_tensor,audit


def record():
    return dict(finite=True,dtype='torch.float32',shape=[4,3],local_shape=[2,3],sha256='a'*64)


def test_exact_geometry_and_scalar_geometry():
    check_tensor(record(),shape=[4,3],local_shape=[2,3])
    scalar=record();scalar.update(shape=[],local_shape=[])
    check_tensor(scalar,shape=[],local_shape=[])


@pytest.mark.parametrize('fault',['finite','dtype','hash','global','local','geometry'])
def test_invalid_hash_or_tensor_contract_rejected(fault):
    r=copy.deepcopy(record())
    if fault=='finite':r['finite']=False
    if fault=='dtype':r['dtype']='torch.bfloat16'
    if fault=='hash':r['sha256']='short'
    if fault=='global':r['shape']=[3,4]
    if fault=='local':r['local_shape']=[4,3]
    if fault=='geometry':r['local_shape']=[True,3]
    with pytest.raises(ValueError):check_tensor(r,shape=[4,3],local_shape=[2,3])


def synthetic_gate(root):
    """Schema/provenance fixture only: not GPU execution or model evidence."""
    shapes={'model.embed_tokens.weight':[151936,2048],'model.norm.weight':[2048]}
    layer={'input_layernorm.weight':[2048],'post_attention_layernorm.weight':[2048],
        'self_attn.q_proj.weight':[2048,2048],'self_attn.k_proj.weight':[1024,2048],
        'self_attn.v_proj.weight':[1024,2048],'self_attn.o_proj.weight':[2048,2048],
        'self_attn.q_norm.weight':[128],'self_attn.k_norm.weight':[128],
        'mlp.gate_proj.weight':[6144,2048],'mlp.up_proj.weight':[6144,2048],
        'mlp.down_proj.weight':[2048,6144]}
    for i in range(28):shapes.update({f'model.layers.{i}.{n}':shape for n,shape in layer.items()})
    def tensor(shape,*,sharded=True,digest='a'*64):
        return dict(shape=shape,local_shape=[shape[0]//2]+shape[1:] if sharded else shape,
            dtype='torch.float32',finite=True,sha256=digest)
    steps=[]
    for step in range(1,4):
        params={n:tensor(s) for n,s in shapes.items()}
        states={n:dict(exp_avg=tensor(s),exp_avg_sq=tensor(s),
            step=tensor([],sharded=False,digest=hashlib.sha256(struct.pack('<f',float(step))).hexdigest()))
            for n,s in shapes.items()}
        steps.append(dict(step=step,loss=2.,input_sha256='e'*64,parameters=params,
            gradients=params,optimizer_states=states,buffers={'model.rotary_emb.inv_freq':tensor([64],sharded=False)},
            param_groups=[dict(lr=1e-5,betas=[.9,.999],eps=1e-8,weight_decay=.01,fused=True)]))
    sources={'files':{'benchmarks/compare_fsdp_training.py':'c'*64}}
    sources['sha256']=hashlib.sha256(json.dumps(sources['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (root/'gate').mkdir();raw_reports=[]
    for rank in (0,1):
        result=dict(passed=True,rank=rank,world=2,deterministic=True,real_parameters=1720574976,
            parameter_entries_per_rank=310,steps_per_phase=3,implementation_sources=sources,worker_sha256='d'*64,
            phases={p:steps for p in ('native_a','native_b','keep28','checkpoint28')})
        path=root/'gate'/f'rank{rank}_result.json';raw=json.dumps(result).encode();path.write_bytes(raw)
        raw_reports.append(dict(rank=rank,path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
    (root/'worker.log').write_text('synthetic fixture, not real execution\n')
    summary=dict(passed=True,world=2,steps=3,real_parameters=1720574976,parameter_entries_per_rank=310,
        raw_reports=raw_reports,worker_sha256='d'*64,frozen_content_sha256='f'*64,
        log_sha256=hashlib.sha256((root/'worker.log').read_bytes()).hexdigest())
    (root/'result.json').write_text(json.dumps(summary))


def test_full_synthetic_all_rank_coverage_audit(tmp_path):
    synthetic_gate(tmp_path)
    report=audit(tmp_path)
    assert report['full_gradients_exact'] and report['full_adamw_states_exact']
    assert not report['convergence_proven'] and not report['defaults_changed']


@pytest.mark.parametrize('fault',['missing_rank','missing_state','nonfinite','bad_counter','bad_local_shape'])
def test_semantic_evidence_faults_rejected_even_with_updated_file_hash(tmp_path,fault):
    synthetic_gate(tmp_path)
    summary_path=tmp_path/'result.json';summary=json.loads(summary_path.read_text())
    if fault=='missing_rank':summary['raw_reports'].pop()
    else:
        path=tmp_path/'gate/rank0_result.json';result=json.loads(path.read_text())
        for phase in result['phases'].values():
            record=phase[0];name='model.norm.weight'
            if fault=='missing_state':record['optimizer_states'][name].pop('exp_avg')
            if fault=='nonfinite':record['gradients'][name]['finite']=False
            if fault=='bad_counter':record['optimizer_states'][name]['step']['sha256']='0'*64
            if fault=='bad_local_shape':record['parameters'][name]['local_shape']=[2048]
        raw=json.dumps(result).encode();path.write_bytes(raw)
        summary['raw_reports'][0]['sha256']=hashlib.sha256(raw).hexdigest()
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):audit(tmp_path)
