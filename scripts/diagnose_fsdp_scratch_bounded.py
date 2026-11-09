"""Real8B four-update native/native/bounded numerical diagnostic, not timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmarks.deterministic_fsdp_scratch_bounded import hashes as worker_hashes
from scripts.audit_fsdp_exchange_confirmation import loss_comparison
from scripts.audit_fsdp_scratch_bounded import audit as audit_performance
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.fsdp_resource_budget import budget_ready
from scripts.screen_fsdp_1p7b import park_waiting_queue
from scripts.summarize_fsdp_memory import audit_result
from training.fsdp_provenance import source_snapshot


def command(candidate, output):
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
           '-m', 'benchmarks.deterministic_fsdp_scratch_bounded', '--backend', 'official',
           '--model', '/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
           '--seq', '512', '--batch', '1', '--warmup', '1', '--steps', '3',
           '--keep-full-layers', '34', '--checkpoint-every', '1', '--memory-budget-gib', '40',
           '--reference-check', '--output', str(output)]
    if candidate:
        cmd += ['--bounded-exchange-scope', 'decoder']
    return cmd


def audit_short(directory, candidate):
    raw = (directory / 'result.json').read_bytes()
    result = json.loads(raw)
    signature = audit_result(result, (34, 1, 'full'), allow_exchange=True)
    opt = result['optimization']
    if (not result['complete'] or result['global_parameters'] != 8190735360
            or result['world_size'] != 4 or result['seq'] != 512 or result['batch'] != 1
            or result['warmup'] != 1 or result['steps'] != 3 or result['memory_budget_gib'] != 40
            or opt['exchange_reduce'] != ('triton' if candidate else None)
            or opt['exchange_scope'] != ('decoder' if candidate else 'all')
            or any(v is not None for v in result['communication_environment'].values())):
        raise ValueError('Wrong real8B diagnostic workload')
    inputs = result['manifest']['input_sha256_by_rank']
    if len(inputs) != 4 or any(len(value) != 64 for value in inputs):
        raise ValueError('Complete ordered rank input hashes required')
    manifests = []
    expected_names = {f'model.layers.{i}' for i in range(36)} if candidate else set()
    for rank in range(4):
        path = directory / f'bounded_manifest_rank{rank}.json'
        wrapper_raw = path.read_bytes()
        wrapper = json.loads(wrapper_raw)
        context_path = directory / f'deterministic_context_rank{rank}.json'
        context_raw = context_path.read_bytes()
        context = json.loads(context_raw)
        counts = result['exchange_callback_counts_by_rank'][rank]
        if (set(counts) != expected_names or any(v != 4 for v in counts.values())
                or wrapper['actual_callback_counts'] != list(counts.values())
                or not wrapper['passed'] or wrapper['rank'] != rank
                or wrapper['actual_adapter'] != ('BoundedScratchExchangeReduceScatter' if candidate else 'native')
                or wrapper['actual_value_validation_enabled'] != candidate
                or wrapper['bounded_scope'] != ('decoder' if candidate else None)
                or any(worker_hashes().get(name) != h for name, h in wrapper['source_sha256'].items())
                or not context['passed'] or context['rank'] != rank
                or context['algorithms_enabled'] is not True or context['warn_only'] is not False
                or context['cublas_workspace'] != ':4096:8'
                or context['source_sha256'] != worker_hashes()
                or context['bounded_manifest_sha256'] != hashlib.sha256(wrapper_raw).hexdigest()):
            raise ValueError('Actual adapter or deterministic settings not proven on every rank')
        if rank == 0 and wrapper['base_result_sha256'] != hashlib.sha256(raw).hexdigest():
            raise ValueError('Wrapper does not bind base result')
        manifests.append(dict(wrapper_sha256=hashlib.sha256(wrapper_raw).hexdigest(),
                              context_sha256=hashlib.sha256(context_raw).hexdigest()))
    return result, signature + (tuple(inputs), json.dumps(worker_hashes(), sort_keys=True)), manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--park-pid', type=int, required=True)
    parser.add_argument('--park-status', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def status(phase, **fields):
        (args.output / 'status.json').write_text(json.dumps(dict(phase=phase, **fields), indent=2))
        print(phase, fields, flush=True)
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    def hashes():
        names = ('scripts/diagnose_fsdp_scratch_bounded.py', 'scripts/audit_fsdp_scratch_bounded.py',
                 'scripts/screen_fsdp_scratch_bounded.py', 'scripts/diagnose_fsdp_1p7b_repeat.py',
                 'scripts/screen_fsdp_1p7b.py', 'scripts/fsdp_resource_budget.py',
                 'scripts/summarize_fsdp_memory.py', 'scripts/freeze_fsdp_reference.py')
        return dict(worker_hashes(), **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names})
    before, common = hashes(), source_snapshot(Path.cwd())
    frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
    def unchanged():
        if hashes() != before or source_snapshot(Path.cwd()) != common:
            raise ValueError('Diagnostic source changed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE', 'PYTORCH_CUDA_ALLOC_CONF'):
        env.pop(key, None)
    try:
        status('verifying_performance_and_inputs')
        performance = audit_performance(Path('artifacts/fsdp_scratch_bounded_budget40_v1'))
        if not performance['decisions']['bounded_decoder_k34']['over_two_percent_both_orders']:
            raise ValueError('Repeated performance candidate not confirmed')
        verify_frozen(frozen); unchanged()
        with park_waiting_queue(args.park_pid, args.park_status):
            runs, rows, signatures = {}, [], set()
            for name, candidate in (('native_a', False), ('native_b', False), ('bounded_decoder', True)):
                status('waiting_for_idle_gpus', case=name)
                samples = []
                while True:
                    values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                                                      '--format=csv,noheader,nounits'], text=True).splitlines()
                    readings = [[int(v.strip()) for v in row.split(',')] for row in values[:4]]
                    samples = samples + [dict(unix_seconds=time.time(), readings=readings)] if budget_ready(
                        readings, allocator_gib=40., free_margin_gib=2.5) else []
                    if len(samples) == 3:
                        (args.output / (name + '_resources.json')).write_text(json.dumps(dict(
                            allocator_budget_gib=40, free_margin_gib=2.5, idle_confirmations=3, samples=samples), indent=2))
                        break
                    time.sleep(15)
                unchanged(); status('running_diagnostic', case=name)
                log = args.output / (name + '.log')
                if run_owned(command(candidate, args.output / name), log, env, timeout=600):
                    raise RuntimeError(f'{name}: diagnostic failed; no fallback or tolerance change')
                unchanged()
                result, signature, manifests = audit_short(args.output / name, candidate)
                if result['manifest']['implementation_sources'] != common:
                    raise ValueError('Worker/common implementation differs')
                signatures.add(signature); runs[name] = result
                path = args.output / name / 'result.json'
                rows.append(dict(case=name, source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), rank_manifests=manifests))
                (args.output / 'progress.json').write_text(json.dumps(rows, indent=2))
            if len(signatures) != 1:
                raise ValueError('Diagnostic inputs/implementations differ')
            native = loss_comparison(runs['native_a'], runs['native_b'])
            candidate = {name: loss_comparison(runs['bounded_decoder'], runs[name]) for name in ('native_a', 'native_b')}
            native_exact = [v['loss'] for v in runs['native_a']['records']] == [v['loss'] for v in runs['native_b']['records']]
            verify_frozen(frozen); unchanged()
            with (args.output / 'result.json').open('x') as file:
                json.dump(dict(complete=True, records=rows, native_repeat=native, native_repeat_exact=native_exact,
                               candidate_comparisons=candidate,
                               eligible_for_quality_preflight=native_exact and all(v['within_declared_loss_tolerance'] for v in candidate.values()),
                               implementation_sources=common, helper_sources=before,
                               frozen_content_sha256=frozen['content_sha256'], default_changed=False,
                               convergence_proven=False, ordinary_performance_claim=False,
                               scope='Four-update real8B deterministic loss diagnostic only; no full gradient/moment or convergence proof'), file, indent=2)
        status('finished', original_42gib_supervisor_resumed=True)
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
