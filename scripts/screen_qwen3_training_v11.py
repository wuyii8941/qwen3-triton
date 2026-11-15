"""Screen training strategies; full gradient/quality qualification is separate."""
from pathlib import Path
import argparse
import json
import os
import shutil
import statistics
import subprocess
import time

from qwen3_triton.gradient_archive import sha
from qwen3_triton.benchmark_utils.gpu_lease import apps
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from qwen3_triton.benchmark_utils.monitor import run
from training.input_manifest import validate
from training.provenance import source_snapshot

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path('/data1/tzh/proj/qwen_operator_lab')


def strategies():
    def row(name, keep, checkpoint, skip=True, head=False, plan=False):
        return dict(name=name, keep_full_layers=keep, checkpoint_every=checkpoint,
                    skip_embedding_backward_gather=skip, retain_lm_head=head, gemm_plan=plan)
    return [row('control', 0, 1, skip=False), row('embedding', 0, 1),
            row('retain8', 8, 1), row('retain16', 16, 1),
            row('retain16_cp2', 16, 2), row('retain12_cp0', 12, 0),
            row('retain16_head_cp2', 16, 2, head=True),
            row('retain16_cp2_gemm', 16, 2, plan=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.is_relative_to(Path('/home/tzh'))
    assert not output.exists()
    assert shutil.disk_usage(output.parent).free > 3 * 2**30
    protocol = ROOT / 'configs/fsdp_8b.json'
    inputs = INPUT_ROOT / 'artifacts/official_qwen3_inputs_v1.json'
    plan = ROOT / 'results/qwen3_triton_v10_gemm_plan_001.json'
    previous = Path.cwd()
    try:
        os.chdir(INPUT_ROOT)
        identity = validate(inputs, protocol, json.loads(protocol.read_text()))
    finally:
        os.chdir(previous)
    reference = INPUT_ROOT / 'artifacts/triton_qwen3_v1/official_8b_paired_quality_ablation_control/hf_reference_r1/result.json'
    assert identity == json.loads(reference.read_text())['input_manifest_sha256']
    output.mkdir()
    files = [Path(__file__), ROOT/'scripts/train_qwen3_training_v11.py',
             ROOT/'scripts/research_python.sh', protocol, inputs, plan,
             *sorted((ROOT/'qwen3_triton').rglob('*.py'))]
    manifest = dict(schema=1, input_identity=identity, roles=strategies(),
                    sources={str(p): sha(p) for p in files}, core_snapshot=source_snapshot(ROOT),
                    quality_steps=1, warmup_steps=2, measure_steps=4,
                    memory_cap_GiB=41.875, minimum_screen_reduction=0.03,
                    full_gradient_quality_qualified=False, automatic_default_change=False,
                    selection='Clean finite 7-update runs under the memory cap, mean/max loss drift below .002/.01 against control, at least 3% lower median; lowest median wins. Screening only.')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    def state(stage):
        value = dict(pid=os.getpid(), stage=stage)
        (output/'state.json').write_text(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    state('waiting_stable_four_GPU_lease')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    rows = []
    with stable_lease([0,1,2,3]):
        for role in manifest['roles']:
            assert not apps()
            for path, digest in manifest['sources'].items():
                assert sha(path) == digest, path
            assert source_snapshot(ROOT) == manifest['core_snapshot']
            name = role['name']
            state('running_'+name)
            command = ['bash', str(ROOT/'scripts/research_python.sh'), '-m', 'torch.distributed.run',
                       '--standalone', '--nproc_per_node=4', str(ROOT/'scripts/train_qwen3_training_v11.py'),
                       '--variant', 'triton_optimized', '--output', str(output/name),
                       '--protocol', str(protocol), '--input-manifest', str(inputs), '--input-root', str(INPUT_ROOT),
                       '--quality-steps', '1', '--warmup-steps', '2', '--measure-steps', '4',
                       '--keep-full-layers', str(role['keep_full_layers']), '--checkpoint-every', str(role['checkpoint_every'])]
            if role['skip_embedding_backward_gather']:
                command.append('--skip-embedding-backward-gather')
            if role['retain_lm_head']:
                command.append('--retain-lm-head')
            if role['gemm_plan']:
                command.extend(['--gemm-plan', str(plan)])
            try:
                with (output/(name+'.log')).open('x') as log:
                    run(command, cwd=ROOT, env=env, stdout=log, monitor=output/(name+'.gpu.json'))
            except subprocess.CalledProcessError:
                log_text = (output/(name+'.log')).read_text()
                assert 'out of memory' in log_text.lower(), 'Unexpected worker failure; inspect logs'
            monitor = json.loads((output/(name+'.gpu.json')).read_text())
            assert not monitor['foreign_process_seen'] and not monitor['observation_errors'], 'Reject resource-contaminated screen'
            owned = set(monitor['known_owned_pids'])
            for _ in range(30):
                live = apps()
                if not live:
                    break
                assert all(pid in owned for gpu, pid in live)
                time.sleep(1)
            assert not apps()
            if monitor['returncode']:
                rows.append(dict(role=role, completed=False, rejected='CUDA out of memory'))
            else:
                result = json.loads((output/name/'result.json').read_text())
                assert result['complete'] and result['screen_only'] and result['input_manifest_sha256'] == identity
                assert len(result['ranks']) == 4
                for rank in result['ranks']:
                    assert rank['updates'] == 7 and rank['parameter_count'] == rank['optimizer_state_count'] == 399
                    assert rank['all_master_grad_moment_finite_fp32'] and not rank['gradient_comparisons']
                samples = [r['wall_seconds'] for r in result['ranks'][0]['rows'] if r['measured']]
                assert len(samples) == 4
                for rank in result['ranks'][1:]:
                    assert samples == [r['wall_seconds'] for r in rank['rows'] if r['measured']]
                rows.append(dict(role=role, completed=True, median_seconds=statistics.median(samples),
                    samples_seconds=samples, global_losses=[r['global_loss'] for r in result['ranks'][0]['rows']],
                    peak_allocated_gib=max(r['measured_training_peak']['allocated_gib'] for r in result['ranks']),
                    peak_reserved_gib=max(r['measured_training_peak']['reserved_gib'] for r in result['ranks']),
                    result_sha256=sha(output/name/'result.json')))
            (output/'progress.json').write_text(json.dumps(dict(rows=rows), indent=2)+'\n')
    assert rows[0]['completed']
    control = rows[0]
    for row in rows:
        if not row['completed']:
            row['eligible_for_formal_comparison'] = False
            continue
        drifts = [abs(a-b) for a,b in zip(row['global_losses'], control['global_losses'])]
        row['reduction_vs_control'] = 1-row['median_seconds']/control['median_seconds']
        row['loss_mean_drift'] = statistics.mean(drifts)
        row['loss_max_drift'] = max(drifts)
        row['eligible_for_formal_comparison'] = (row['reduction_vs_control'] >= .03 and statistics.mean(drifts) < .002
            and max(drifts) < .01 and max(row['peak_allocated_gib'], row['peak_reserved_gib']) <= 41.875)
    candidates = [r for r in rows[1:] if r['eligible_for_formal_comparison']]
    selected = min(candidates, key=lambda r:r['median_seconds'])['role'] if candidates else None
    report = dict(complete=True, rows=rows, selected_for_formal_comparison=selected,
                  full_gradient_quality_qualified=False, automatic_default_change=False)
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    state('complete_screen_only')
    print(json.dumps(dict(selected=selected)), flush=True)


if __name__ == '__main__':
    main()
