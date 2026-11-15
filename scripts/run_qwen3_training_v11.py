"""Qualify a screened strategy with matching HF and opposite-order controls.

Screening selects candidates; native memory feasibility chooses the first fitting
strategy in that frozen ranking. Full timing and quality do not affect selection.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.gpu_lease import apps
from qwen3_triton.benchmark_utils.monitor import run
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from training.provenance import source_snapshot

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path('/data1/tzh/proj/qwen_operator_lab')
REFERENCE = INPUT_ROOT/'artifacts/triton_qwen3_v1/official_8b_paired_quality_ablation_control/hf_reference_r1'


def roles(selected, baseline):
    return [dict(name=f'{arm}_r{order}', arm=arm, order=order,
                 variant='hf' if arm == 'hf' else 'triton_optimized',
                 strategy=baseline if arm == 'baseline' else selected)
            for order, arms in ((1,('baseline','hf','candidate')),(2,('candidate','hf','baseline')))
            for arm in arms]


def command(output, role, plan, full=True):
    strategy = role['strategy']
    cmd = ['bash', str(ROOT/'scripts/research_python.sh'), '-m', 'torch.distributed.run',
           '--standalone', '--nproc_per_node=4', str(ROOT/'scripts/train_qwen3_training_v11.py'),
           '--variant', role['variant'], '--output', str(output/role['name']),
           '--protocol', str(ROOT/'configs/fsdp_8b.json'),
           '--input-manifest', str(INPUT_ROOT/'artifacts/official_qwen3_inputs_v1.json'),
           '--input-root', str(INPUT_ROOT), '--keep-full-layers', str(strategy['keep_full_layers']),
           '--checkpoint-every', str(strategy['checkpoint_every']),
           '--quality-steps', '16' if full else '1', '--warmup-steps', '3' if full else '2',
           '--measure-steps', '8' if full else '4']
    if full:
        cmd.extend(['--full-validation','--trace','--reference-raw-dir',str(REFERENCE)])
    if strategy['skip_embedding_backward_gather']:
        cmd.append('--skip-embedding-backward-gather')
    if strategy['retain_lm_head']:
        cmd.append('--retain-lm-head')
    if strategy['gemm_plan'] and role['variant'] != 'hf':
        cmd.extend(['--gemm-plan',str(plan)])
    return cmd


def monitored_role(output, role, plan, env, full):
    name = role['name']
    try:
        with (output/(name+'.log')).open('x') as log:
            run(command(output,role,plan,full), cwd=ROOT, env=env, stdout=log,
                monitor=output/(name+'.gpu.json'))
    except subprocess.CalledProcessError:
        assert not full and 'out of memory' in (output/(name+'.log')).read_text().lower(), 'Unexpected or formal worker failure'
    monitor = json.loads((output/(name+'.gpu.json')).read_text())
    assert not monitor['foreign_process_seen'] and not monitor['observation_errors'], 'Reject contaminated role'
    owned = set(monitor['known_owned_pids'])
    for _ in range(30):
        live = apps()
        if not live:
            break
        assert all(pid in owned for gpu,pid in live)
        time.sleep(1)
    assert not apps()
    if monitor['returncode']:
        return None
    result = json.loads((output/name/'result.json').read_text())
    assert result['complete'] and len(result['ranks']) == 4
    assert result['variant'] == role['variant'] and result['model'] == 'Qwen/Qwen3-8B' and result['layers'] == 36
    assert result['screen_only'] == (not full)
    assert all(r['updates'] == (27 if full else 7) and r['parameter_count'] == r['optimizer_state_count'] == 399
               and r['all_master_grad_moment_finite_fp32'] for r in result['ranks'])
    assert all(max(r['measured_training_peak'].values()) <= 41.875 for r in result['ranks'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    screen, output = args.screen.resolve(), args.output.resolve()
    assert not output.is_relative_to(Path('/home/tzh')) and not output.exists()
    screening = json.loads((screen/'result.json').read_text())
    screening_manifest = json.loads((screen/'manifest.json').read_text())
    assert screening['complete'] and not screening['full_gradient_quality_qualified']
    assert screening['selected_for_formal_comparison'] is not None
    ranking = sorted([r for r in screening['rows'][1:] if r['eligible_for_formal_comparison']], key=lambda r:r['median_seconds'])
    assert ranking and ranking[0]['role'] == screening['selected_for_formal_comparison']
    for path,digest in screening_manifest['sources'].items():
        assert sha(path) == digest, path
    assert source_snapshot(ROOT) == screening_manifest['core_snapshot']
    output.mkdir()
    plan = ROOT/'results/qwen3_triton_v10_gemm_plan_001.json'
    files = [Path(__file__), ROOT/'scripts/train_qwen3_training_v11.py',
             ROOT/'scripts/qwen3_training_v11_contract.py', ROOT/'scripts/review_qwen3_training_v11.py',
             ROOT/'scripts/research_python.sh', screen/'result.json', screen/'manifest.json',
             ROOT/'configs/fsdp_8b.json', INPUT_ROOT/'artifacts/official_qwen3_inputs_v1.json',
             REFERENCE/'result.json', plan, *sorted((ROOT/'qwen3_triton').rglob('*.py'))]
    sources = {str(p):sha(p) for p in files}
    selection = dict(complete=True, ranking=[r['role'] for r in ranking], sources=sources,
        core_snapshot=source_snapshot(ROOT), rule='Choose first nativeHF memory-feasible strategy in screened median ranking; no formal observations used.',
        minimum_reduction_each_order=.03, memory_cap_GiB=41.875, default_changed=False)
    (output/'selection_manifest.json').write_text(json.dumps(selection,indent=2)+'\n')
    def verify():
        for path,digest in sources.items():
            assert sha(path) == digest, path
        assert source_snapshot(ROOT) == selection['core_snapshot']
    def state(stage):
        value = dict(pid=os.getpid(), stage=stage)
        (output/'state.json').write_text(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    state('waiting_stable_four_GPU_lease')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    feasibility = []
    with stable_lease([0,1,2,3]):
        selected = None
        for row in ranking:
            verify()
            strategy = row['role']
            role = dict(name='native_feasibility_'+strategy['name'], variant='hf', strategy=strategy)
            state(role['name'])
            result = monitored_role(output,role,plan,env,False)
            feasibility.append(dict(name=role['name'], strategy=strategy, fits=result is not None,
                monitor_sha256=sha(output/(role['name']+'.gpu.json')),
                log_sha256=sha(output/(role['name']+'.log')),
                result_sha256=sha(output/role['name']/'result.json') if result else None))
            if result is not None:
                selected = strategy
                break
        (output/'native_feasibility.json').write_text(json.dumps(feasibility,indent=2)+'\n')
        if selected is None:
            state('no_native_feasible_candidate')
            return
        queue = roles(selected,screening_manifest['roles'][0])
        manifest = dict(schema=1, sources=sources, selected_strategy=selected, roles=queue,
            input_identity=screening_manifest['input_identity'], input_root=str(INPUT_ROOT),
            reference_dir=str(REFERENCE), reference_result_sha256=sha(REFERENCE/'result.json'),
            gemm_plan=dict(path=str(plan),sha256=sha(plan)), minimum_reduction_each_order=.03,
            memory_cap_GiB=41.875, selection_manifest_sha256=sha(output/'selection_manifest.json'),
            native_feasibility_sha256=sha(output/'native_feasibility.json'), performance_adopted=False)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for role in queue:
            verify()
            assert not apps()
            state('running_'+role['name'])
            result = monitored_role(output,role,plan,env,True)
            assert result is not None
        verify()
    state('GPU_complete_independent_CPU_review')
    subprocess.run(['bash',str(ROOT/'scripts/research_python.sh'),str(ROOT/'scripts/review_qwen3_training_v11.py'),
                    '--output',str(output)],cwd=ROOT,check=True)
    state('complete_reviewed_no_default_change')


if __name__ == '__main__':
    main()
