"""Follow up a completed strategy screen with a frozen explicit profile list."""
from pathlib import Path
import argparse
import json
import os
import statistics

from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.gpu_lease import apps
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from training.provenance import source_snapshot
from run_qwen3_training_v11 import monitored_role

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path('/data1/tzh/proj/qwen_operator_lab')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-screen', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent, profiles_path, output = args.parent_screen.resolve(), args.profiles.resolve(), args.output.resolve()
    assert not output.is_relative_to(Path('/home/tzh')) and not output.exists()
    previous = json.loads((parent/'result.json').read_text())
    original_manifest = json.loads((parent/'manifest.json').read_text())
    assert previous['complete'] and previous['selected_for_formal_comparison'] is not None
    profile_record = json.loads(profiles_path.read_text())
    assert profile_record['parent_result_sha256'] == sha(parent/'result.json')
    planner = Path(profile_record['planner_source'])
    assert sha(planner) == profile_record['planner_source_sha256']
    queue = profile_record['roles']
    assert queue[0] == dict(name='control',keep_full_layers=0,checkpoint_every=1,
        skip_embedding_backward_gather=False,retain_lm_head=False,gemm_plan=False)
    assert len({r['name'] for r in queue}) == len(queue)
    for role in queue:
        assert role['name'].replace('_','').isalnum()
        assert type(role['keep_full_layers']) is int and 0 <= role['keep_full_layers'] <= 36
        assert type(role['checkpoint_every']) is int and role['checkpoint_every'] >= 0
        assert all(type(role[k]) is bool for k in ['skip_embedding_backward_gather','retain_lm_head','gemm_plan'])
    assert source_snapshot(ROOT) == original_manifest['core_snapshot']
    for path,digest in original_manifest['sources'].items():
        assert sha(path) == digest, path
    output.mkdir()
    plan = ROOT/'results/qwen3_triton_v10_gemm_plan_001.json'
    files = [Path(__file__),ROOT/'scripts/run_qwen3_training_v11.py',ROOT/'scripts/train_qwen3_training_v11.py',
             ROOT/'scripts/research_python.sh',planner,profiles_path,parent/'result.json',parent/'manifest.json',plan,
             ROOT/'configs/fsdp_8b.json',INPUT_ROOT/'artifacts/official_qwen3_inputs_v1.json',
             *sorted((ROOT/'qwen3_triton').rglob('*.py'))]
    manifest = dict(schema=1,sources={str(p):sha(p) for p in files},core_snapshot=source_snapshot(ROOT),
        roles=queue,input_identity=original_manifest['input_identity'],quality_steps=1,warmup_steps=2,measure_steps=4,
        memory_cap_GiB=41.875,minimum_screen_reduction=.03,full_gradient_quality_qualified=False,
        selection='Retest control/incumbent and higher-retention candidates; >=3% reduction with .002/.01 mean/max loss drift and unchanged memory cap. Lowest eligible median wins.',
        parent_screen=str(parent),profiles_sha256=sha(profiles_path),automatic_default_change=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def state(stage):
        value = dict(pid=os.getpid(),stage=stage)
        (output/'state.json').write_text(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    state('waiting_stable_four_GPU_lease')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    rows = []
    with stable_lease([0,1,2,3]):
        for strategy in queue:
            for path,digest in manifest['sources'].items():
                assert sha(path) == digest, path
            assert source_snapshot(ROOT) == manifest['core_snapshot'] and not apps()
            state('running_'+strategy['name'])
            role = dict(name=strategy['name'],variant='triton_optimized',strategy=strategy)
            result = monitored_role(output,role,plan,env,False)
            if result is None:
                rows.append(dict(role=strategy,completed=False,rejected='CUDA out of memory'))
            else:
                assert result['input_manifest_sha256'] == manifest['input_identity']
                assert all(not r['gradient_comparisons'] for r in result['ranks'])
                samples = [r['wall_seconds'] for r in result['ranks'][0]['rows'] if r['measured']]
                assert len(samples) == 4 and all(samples == [r['wall_seconds'] for r in rank['rows'] if r['measured']] for rank in result['ranks'])
                rows.append(dict(role=strategy,completed=True,median_seconds=statistics.median(samples),samples_seconds=samples,
                    global_losses=[r['global_loss'] for r in result['ranks'][0]['rows']],
                    peak_allocated_gib=max(r['measured_training_peak']['allocated_gib'] for r in result['ranks']),
                    peak_reserved_gib=max(r['measured_training_peak']['reserved_gib'] for r in result['ranks']),
                    result_sha256=sha(output/strategy['name']/'result.json')))
            (output/'progress.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    assert rows[0]['completed']
    for row in rows:
        if not row['completed']:
            row['eligible_for_formal_comparison'] = False
            continue
        drift = [abs(a-b) for a,b in zip(row['global_losses'],rows[0]['global_losses'])]
        row['reduction_vs_control'] = 1-row['median_seconds']/rows[0]['median_seconds']
        row['loss_mean_drift'],row['loss_max_drift'] = statistics.mean(drift),max(drift)
        row['eligible_for_formal_comparison'] = (row['reduction_vs_control'] >= .03 and statistics.mean(drift)<.002
            and max(drift)<.01 and max(row['peak_allocated_gib'],row['peak_reserved_gib'])<=41.875)
    eligible = [r for r in rows[1:] if r['eligible_for_formal_comparison']]
    winner = min(eligible,key=lambda r:r['median_seconds'])['role'] if eligible else None
    report = dict(complete=True,rows=rows,selected_for_formal_comparison=winner,
        full_gradient_quality_qualified=False,automatic_default_change=False)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    state('complete_screen_only')
    print(json.dumps(dict(selected=winner)),flush=True)


if __name__ == '__main__':
    main()
