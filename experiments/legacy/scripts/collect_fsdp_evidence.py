"""Audit completed real-8B deployment and emit only small numeric evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import triton
import transformers
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('artifacts/fsdp_8b_deployment_v1'))
    parser.add_argument('--output',type=Path,default=Path('results/fsdp_8b_deployment.json'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    phase = json.loads((args.run/'pipeline_status.json').read_text())
    check = json.loads((args.run/'verification.json').read_text())
    ranks = [json.loads((args.run/f'rank{i}.json').read_text()) for i in range(4)]
    assert phase['phase'] == 'verified' and check['passed'] and check['generated_tokens'] > 0
    for rank in ranks:
        assert rank['completed'] and not rank['random_init'] and rank['full_parameter_training']
        assert rank['world_size'] == 4 and rank['global_parameters'] >= 7_000_000_000
        assert rank['initial_hf_parity']['max_abs'] == 0 and len(rank['records']) >= 3
        assert rank['data'] and rank['seq'] == 512 and rank['batch'] == 1
    assert sum(rank['local_parameters'] for rank in ranks) == ranks[0]['global_parameters']
    exported = args.run/'hf_model'
    index = json.loads((exported/'model.safetensors.index.json').read_text())
    assert all((exported/name).is_file() for name in set(index['weight_map'].values()))
    key = 'model.layers.0.self_attn.q_proj.weight'
    source = Path(ranks[0]['model'])
    original_index = json.loads((source/'model.safetensors.index.json').read_text())
    with safe_open(str(source/original_index['weight_map'][key]),framework='pt') as stream:
        before = stream.get_tensor(key)
    with safe_open(str(exported/index['weight_map'][key]),framework='pt') as stream:
        after = stream.get_tensor(key)
    changed = (before != after).float().mean().item()
    assert changed > 0, 'No actual weight updates survived BF16 export'
    source_files = ['training/minimal_fsdp.py','training/block_fsdp.py',
        'training/run_sharded_qwen.py','training/sharded_checkpoint.py',
        'training/export_sharded_hf.py','scripts/deploy_sharded_8b.py',
        'benchmarks/verify_exported_qwen.py']
    result = dict(verified=True,checkpoint=source.name,model_type='qwen3',
        global_parameters=ranks[0]['global_parameters'],world_size=4,gpu='RTX A6000',
        dtype='BF16 compute; FP32 master/gradients/AdamW',seq=512,microbatch_per_rank=1,
        global_batch=4,steps=len(ranks[0]['records']),data='Existing WikiText token corpus',
        full_parameter_training=True,initial_hf_max_abs=0.,export_reload_max_abs=check['max_abs'],
        greedy_generated_tokens=check['generated_tokens'],
        max_peak_allocated_gib=max(s['peak_allocated_gib'] for rank in ranks for s in rank['records']),
        max_peak_reserved_gib=max(s['peak_reserved_gib'] for rank in ranks for s in rank['records']),
        q_projection_changed_fraction=changed,
        q_projection_max_abs_update=(before.float()-after.float()).abs().max().item(),
        pytorch=torch.__version__,triton=triton.__version__,transformers=transformers.__version__,
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files},
        limits=['Short training smoke, not convergence or quality evaluation',
                'Not a benchmark against official FSDP',
                'HF model export verified; full 8B AdamW checkpoint not saved due disk capacity',
                'Original Qwen/Qwen3-8B download incomplete; real checkpoint is DeepSeek Qwen3 derivative',
                'YaRN attn_factor warning retained under HF 4.51.0; long context not validated'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
