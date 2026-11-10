"""Fresh-wrapper real8B40GiB ABCD/DCBA; original42GiB study is only parked."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmarks.compare_fsdp_scratch_bounded import sources
from scripts.audit_fsdp_exchange_confirmation import callback_contract
from scripts.confirm_fsdp_exchange import audit_case
from scripts.diagnose_fsdp_1p7b_repeat import run_owned
from scripts.freeze_fsdp_reference import verify_frozen
from scripts.fsdp_resource_budget import budget_ready
from scripts.screen_fsdp_1p7b import park_waiting_queue
from training.fsdp_provenance import source_snapshot


CASES = {'native_k32': (32, None), 'native_k34': (34, None),
         'bounded_head_k34': (34, 'head'), 'bounded_decoder_k34': (34, 'decoder')}


def command(spec, output):
    keep, scope = spec
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
           '-m', 'benchmarks.compare_fsdp_scratch_bounded', '--backend', 'official',
           '--model', '/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
           '--seq', '512', '--batch', '1', '--warmup', '3', '--steps', '12',
           '--keep-full-layers', str(keep), '--checkpoint-every', '1',
           '--memory-budget-gib', '40', '--reference-check', '--output', str(output)]
    if scope:
        cmd += ['--bounded-exchange-scope', scope]
    return cmd


def audit_run(directory, spec):
    keep, scope = spec
    raw = (directory / 'result.json').read_bytes()
    result = json.loads(raw)
    contract = (keep, 'triton' if scope else None, scope or 'all', None)
    signature = audit_case(result, contract)
    callback_contract(result, contract)
    if not result['complete'] or result['memory_budget_gib'] != 40:
        raise ValueError('Complete real8B40GiB run required')
    wrappers = []
    for rank in range(4):
        path = directory / f'bounded_manifest_rank{rank}.json'
        data = path.read_bytes()
        row = json.loads(data)
        if (not row['passed'] or row['rank'] != rank or row['bounded_scope'] != scope
                or row['actual_adapter'] != ('BoundedScratchExchangeReduceScatter' if scope else 'native')
                or row['actual_value_validation_enabled'] != bool(scope)
                or row['input_mutation'] != bool(scope)
                or row['legacy_base_debug_contract_applicable'] != (not bool(scope))
                or row['actual_callback_counts'] != list(result['exchange_callback_counts_by_rank'][rank].values())
                or row['source_sha256'] != sources()):
            raise ValueError('Actual wrapper adapter/source/validation evidence mismatch')
        if rank == 0 and row['base_result_sha256'] != hashlib.sha256(raw).hexdigest():
            raise ValueError('Unbound base result')
        wrappers.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest()))
    return result, signature + (json.dumps(sources(), sort_keys=True),), wrappers


def verify_gates():
    gate_path = Path('artifacts/fsdp_scratch_bounded_model_gate_v4/result.json')
    gate = json.loads(gate_path.read_text())
    if (gate.get('passed') is not True or gate['actual_adapter'] != 'BoundedScratchExchangeReduceScatter'
            or len(gate['rank_records']) != 4 or len(gate['gates']) != 2
            or {row['tied'] for row in gate['gates']} != {True, False}):
        raise ValueError('Complete bounded D128 tied/untied gate required')
    for row in gate['gates']:
        if (not row['passed'] or row['world'] != 4 or row['head_dim'] != 128 or row['seq'] != 129
                or row['steps'] != 3 or row['checkpoint_every'] != 1 or not row['deterministic']):
            raise ValueError('Wrong numerical gate workload')
    if hashlib.sha256((gate_path.parent / 'workers.log').read_bytes()).hexdigest() != gate['log_sha256']:
        raise ValueError('Numerical gate log changed')
    for name, digest in gate['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('Numerically tested adapter source changed')
    for rank, row in enumerate(gate['rank_records']):
        if (row != json.loads((gate_path.parent / f'rank{rank}.json').read_text())
                or row['rank'] != rank or not row['passed'] or sum(row['callback_calls']) != 27):
            raise ValueError('Missing numerical callback evidence')
    smoke_path = Path('artifacts/fsdp_scratch_bounded_wrapper_smoke_v2/result.json')
    smoke = json.loads(smoke_path.read_text())
    if (not smoke['passed'] or smoke['source_sha256'] != sources()
            or [row['case'] for row in smoke['records']] != ['native', 'head', 'decoder']):
        raise ValueError('Matching wrapper GPU smoke required')
    for row in smoke['records']:
        raw_path = Path(row['source'])
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Smoke raw result changed')
        if hashlib.sha256((smoke_path.parent / (row['case'] + '.log')).read_bytes()).hexdigest() != row['log_sha256']:
            raise ValueError('Smoke log changed')
        for rank, manifest in enumerate(row['actual_manifests']):
            if (manifest != json.loads((raw_path.parent / f'bounded_manifest_rank{rank}.json').read_text())
                    or not manifest['passed'] or manifest['source_sha256'] != sources()):
                raise ValueError('Smoke wrapper manifest changed')
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (gate_path, smoke_path)}


def decisions(rows):
    indexed = {(row['repeat'], row['case']): row for row in rows}
    output = {}
    for name in ('bounded_head_k34', 'bounded_decoder_k34'):
        ratios = []
        for repeat in (0, 1):
            candidate = indexed[(repeat, name)]
            controls = [indexed[(repeat, n)] for n in ('native_k32', 'native_k34')
                        if indexed[(repeat, n)]['status'] == 'passed']
            ratios.append(min(v['median_step_seconds'] for v in controls) / candidate['median_step_seconds']
                          if candidate['status'] == 'passed' and controls else None)
        output[name] = dict(ratios_vs_best_tested_native=ratios,
                            over_two_percent_both_orders=all(v is not None and v > 1.02 for v in ratios),
                            convergence_proven=False, default_changed=False)
    return output


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
    common = source_snapshot(Path.cwd())
    helper_files = ('scripts/screen_fsdp_scratch_bounded.py', 'scripts/confirm_fsdp_exchange.py',
                    'scripts/audit_fsdp_exchange_confirmation.py', 'scripts/fsdp_resource_budget.py',
                    'scripts/screen_fsdp_1p7b.py', 'scripts/diagnose_fsdp_1p7b_repeat.py',
                    'scripts/freeze_fsdp_reference.py', 'scripts/summarize_fsdp_memory.py')
    def hashes():
        return dict(sources(), **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in helper_files})
    before = hashes()
    def unchanged():
        if hashes() != before or source_snapshot(Path.cwd()) != common:
            raise ValueError('Study implementation changed')
    frozen = json.loads(Path('results/frozen_fsdp_reference_v1.json').read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False')
    for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE',
                'CUBLAS_WORKSPACE_CONFIG', 'PYTORCH_CUDA_ALLOC_CONF'):
        env.pop(key, None)
    def idle(label):
        status('waiting_for_idle_gpus', case=label, allocator_budget_gib=40, free_margin_gib=2.5)
        samples = []
        while True:
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                                            '--format=csv,noheader,nounits'], text=True).splitlines()
            values = [[int(v.strip()) for v in row.split(',')] for row in rows[:4]]
            samples = samples + [dict(unix_seconds=time.time(), readings=values)] if budget_ready(
                values, allocator_gib=40., free_margin_gib=2.5) else []
            if len(samples) == 3:
                (args.output / (label + '_resources.json')).write_text(json.dumps(dict(
                    allocator_budget_gib=40, free_margin_gib=2.5, idle_confirmations=3, samples=samples), indent=2))
                return
            time.sleep(15)
    try:
        status('verifying_frozen_inputs_and_gates')
        gate_hashes = verify_gates()
        verify_frozen(frozen)
        unchanged()
        with park_waiting_queue(args.park_pid, args.park_status):
            rows, signatures = [], set()
            for repeat, order in ((0, list(CASES)), (1, list(reversed(CASES)))):
                for name in order:
                    label = f'r{repeat}_{name}'
                    idle(label); unchanged(); status('running_model', case=label)
                    log = args.output / (label + '.log')
                    code = run_owned(command(CASES[name], args.output / label), log, env, timeout=3600)
                    unchanged()
                    row = dict(repeat=repeat, case=name, returncode=code,
                               log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
                    if code:
                        if not any(token in log.read_bytes() for token in
                                   (b'CUDA out of memory', b'torch.OutOfMemoryError', b'torch.cuda.OutOfMemoryError')):
                            raise RuntimeError(f'{label}: non-capacity failure')
                        row['status'] = 'oom'
                    else:
                        directory = args.output / label
                        result, signature, manifests = audit_run(directory, CASES[name])
                        signatures.add(signature)
                        row.update(status='passed', source=str(directory / 'result.json'),
                                   sha256=hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest(),
                                   actual_manifests=manifests, median_step_seconds=result['median_step_seconds'],
                                   peak_allocated_gib=max(v['peak_allocated_gib'] for v in result['records']),
                                   peak_reserved_gib=max(v['peak_reserved_gib'] for v in result['records']))
                    rows.append(row)
                    (args.output / 'progress.json').write_text(json.dumps(rows, indent=2))
            if len(signatures) != 1 or any(row['status'] != 'passed' for row in rows if row['case'] == 'native_k32'):
                raise ValueError('Matched passing native controls required')
            verify_frozen(frozen); unchanged()
            if verify_gates() != gate_hashes:
                raise ValueError('Gate evidence changed')
            with (args.output / 'result.json').open('x') as file:
                json.dump(dict(complete=True, records=rows, decisions=decisions(rows), case_specs=CASES,
                               implementation_sources=common, helper_sources=before, gate_sha256=gate_hashes,
                               frozen_content_sha256=frozen['content_sha256'], memory_budget_gib=40,
                               default_changed=False, original_42gib_study_replaced=False,
                               scope='Bounded original-BF16 exchange real8B40GiB matched wrapper study; not convergence'), file, indent=2)
        status('finished', original_42gib_supervisor_resumed=True)
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
