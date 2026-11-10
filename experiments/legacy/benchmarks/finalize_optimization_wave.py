"""Generate one evidence index only after every finite experiment finished."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


def read(name):
    return json.loads((Path('artifacts')/name).read_text())


def main():
    required=['optimization_matrix_v1/progress.json','all_tensor_flashoptim_v1/result.json',
              'all_tensor_flashoptim_vectorized_v1/result.json','all_tensor_flashoptim_async_v1/result.json',
              'graphed_training_v1/result.json','graphed_training_unrestricted_v1/result.json',
              'graphed_training_b1_s2048_v1/result.json','embedding_memory_lifetime_v1/result.json',
              'embedding_memory_lifetime_s512_v1/result.json','embedding_memory_lifetime_s512_confirm_v1/result.json']
    for name in required:
        if not read(name).get('complete'):raise RuntimeError(f'Not complete: {name}')
    pairs=[]
    for name in ('embedding_memory_lifetime_s512_v1/result.json','embedding_memory_lifetime_s512_confirm_v1/result.json'):
        rows=read(name)['rows']
        baseline=next(r for r in rows if r['stage']=='baseline')
        candidate=next(r for r in rows if r['stage']=='consume')
        pairs.append(dict(baseline_ms=baseline['median_step_ms'],candidate_ms=candidate['median_step_ms'],
            time_reduction=1-candidate['median_step_ms']/baseline['median_step_ms'],
            baseline_peak_gib=baseline['peak_allocated_bytes']/2**30,candidate_peak_gib=candidate['peak_allocated_bytes']/2**30,
            peak_saved_mib=(baseline['peak_allocated_bytes']-candidate['peak_allocated_bytes'])/2**20,
            max_training_loss_abs=max(abs(a-b) for a,b in zip(baseline['losses'],candidate['losses']))))
    curve=read('consumed_embedding_trajectory_v1/comparison.json')
    gate=read('tied_embedding_consume_s512_gate_v1/result.json')
    tests=ET.parse('artifacts/systematic_optimization_release_v1.xml').getroot().find('testsuite').attrib
    success=bool(gate['passed'] and curve['passed'] and tests['failures']=='0' and tests['errors']=='0'
                 and all(p['time_reduction']>.01 and p['max_training_loss_abs']<=.01 for p in pairs))
    sources=[Path(p) for p in ('training/tied_embedding.py','training/graphed_forward_backward.py',
        'training/multi_tensor_flashoptim.py','training/optimization_policy.py','training/flashoptim_convergence_job.py',
        'benchmarks/benchmark_tied_embedding_step.py','benchmarks/check_graphed_qwen.py')]
    report=dict(finished_time=time.time(),finite_wave_complete=True,
        default_path_unchanged=True,new_large_checkpoints=False,convergence_claim=False,
        recommendation='compile + embedding_consumed for B1/S512 only' if success else 'No adoption: inspect failed gate',
        recommendation_gate=success,consumed_embedding_pairs=pairs,short_trajectory=curve,
        tests={k:tests[k] for k in ('tests','failures','errors','skipped','time')},
        evidence_files=required,
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','transformers','flashoptim')},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        model_identity=subprocess.check_output(['sha256sum',
            '/data1/tzh/models/Qwen/Qwen3-1.7B/config.json',
            '/data1/tzh/models/Qwen/Qwen3-1.7B/model-00001-of-00002.safetensors',
            '/data1/tzh/models/Qwen/Qwen3-1.7B/model-00002-of-00002.safetensors'],text=True).splitlines(),
        gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.used,utilization.gpu',
            '--format=csv'],text=True).splitlines(),
        limitations=['Shared host, unlocked clocks; no new convergence or multi-seed claim.',
                     'S2048 embedding memory regresses; Graph has no reliable long-sequence benefit.',
                     'Graph extra gain shrinks under unrestricted stronger baseline; not default.',
                     'No hardware DRAM counters: ERR_NVGPUCTRPERM, no driver permission changes.'])
    path=Path('artifacts/optimization_wave_final_v1.json')
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(recommendation=report['recommendation'],pairs=pairs,tests=report['tests']),indent=2))


if __name__=='__main__':main()
