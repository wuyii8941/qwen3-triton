"""Export a small local source candidate, excluding data/weights/artifact sources.

This is a reviewable source copy, not a release or publication. Source
licensing and numerical/performance acceptance remain separate. It preserves
the complete public model and the common data/FSDP helpers byte-for-byte.
"""
from pathlib import Path
import argparse,hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]

SCRIPTS=(
    'research_python.sh','train_qwen3_triton_8b.py','run_qwen3_triton_suite.py',
    'review_qwen3_triton_suite.py','summarize_qwen3_triton_suite.py','check_qwen3_triton.py',
    'prepare_qwen3_triton_corpus.py','configure_qwen3_triton_inputs.py',
    'benchmark_qwen3_triton_modules.py','run_qwen3_triton_module_benchmark.py',
    'review_qwen3_triton_modules.py','export_qwen3_triton_candidate.py',
)
COMMON=(
    'training/__init__.py','training/data.py','training/evaluation.py',
    'training/input_manifest.py','training/provenance.py',
    'training/fsdp/__init__.py','training/fsdp/official.py',
    'training/fsdp/activation_checkpoint.py','training/fsdp/vocab_head.py',
    'training/fsdp/vocab_embedding.py','ops/__init__.py','ops/triton/__init__.py',
    'ops/triton/gradient_transport.py','configs/fsdp_8b.json',
)
RESULTS=(
    'qwen3_triton_source_port_v2.json','qwen3_triton_input_recipe_v1.json',
    'qwen3_triton_benchmark_source_port_v2.json','qwen3_triton_module_source_port_v1.json',
    'qwen3_triton_module_review_port_v1.json','qwen3_triton_public_source_check_v2.json',
    'qwen3_triton_source_AST_check_v2.json','qwen3_triton_benchmark_review_port_v2.json',
    'qwen3_triton_corpus_replay_v1.json','qwen3_triton_v9_forward_qualification_v1.json',
    'qwen3_triton_v9_gradient_math_qualification_v1.json','qwen3_triton_v9_fsdp_qualification_v1.json',
    'qwen3_triton_v9_module_benchmark_v1.json','qwen3_triton_public_training_prefix_v2.json',
    'qwen3_triton_public_first_order_v1.json',
    'qwen3_triton_v9_profile_screen_v1.json',
    'qwen3_triton_portable_clone_cpu_v2.json',
)
OPTIONAL_RESULTS=(
    'qwen3_triton_v9_paired_training_v1.json',
    'qwen3_triton_fresh_reference_review_v1.json',
    'qwen3_triton_portable_clone_cpu_v3.json',
    'qwen3_triton_v9_data_acceptance_v1.json',
)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def paired_evidence():
    path=ROOT/'results/qwen3_triton_v9_paired_training_v1.json'
    if not path.exists():return dict(numerical_training_contract_qualified=False,paired_report=None)
    report=json.loads(path.read_text())
    assert report['complete'] and report['paired_quality_and_coverage_passed']
    assert report['model']=='Qwen/Qwen3-8B' and report['layers']==36 and report['world_size']==4
    assert report['parameter_tensors_per_rank']==399 and report['actual_updates_per_role']==27
    assert len(report['comparisons'])==8 and all([row['order'] for row in variant['orders']]==[1,2] for variant in report['comparisons'])
    result=dict(numerical_training_contract_qualified=True,
                paired_report=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),
                                   interrupted_cohort=report['interrupted_cohort']))
    accepted=ROOT/'results/qwen3_triton_v9_data_acceptance_v1.json'
    if accepted.exists():
        proof=json.loads(accepted.read_text())
        assert proof['complete'] and proof['agent_data_and_runtime_audit_passed']
        result['agent_data_runtime_acceptance']=dict(path=str(accepted.relative_to(ROOT)),sha256=sha(accepted))
    return result

def selected_files():
    package=[p.relative_to(ROOT) for p in sorted((ROOT/'qwen3_triton').rglob('*'))
        if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts]
    files=package+[Path('scripts')/name for name in SCRIPTS]+list(map(Path,COMMON))
    files += [Path('results')/name for name in RESULTS]
    files += [Path('results')/name for name in OPTIONAL_RESULTS if (ROOT/'results'/name).is_file()]
    files += list(map(Path,('TRITON_QWEN3.md','QWEN3_TRITON_REPRODUCE.md','requirements-qwen3-triton.txt')))
    assert len(files)==len(set(files))
    for rel in files:
        path=ROOT/rel
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        assert not any(part in ('artifacts','.runtime-research','.venv-research','__pycache__') for part in rel.parts)
    return sorted(files)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();files=selected_files();qualification=paired_evidence()
    if args.plan:
        print(json.dumps(dict(source_files=len(files),source_bytes=sum((ROOT/p).stat().st_size for p in files),published=False,qualification=qualification,files=list(map(str,files))),indent=2));return
    output=args.output.resolve()
    if output.is_relative_to(Path('/home/tzh')):parser.error('Output under /home/tzh is forbidden')
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for rel in files:
        source=ROOT/rel;destination=output/rel;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        assert sha(source)==sha(destination)
        rows.append(dict(path=str(rel),sha256=sha(source),bytes=source.stat().st_size))
    readme='''# Complete Triton Qwen3-8B source candidate

This is a local source export for review, not a published or licensed release.
The V9 source passes full8B BF16 forward and four-rank gradient/state and
CUDA-coverage checks. The public HF/base/optimized entries each complete
27 updates and full validation before/after16updates within the frozen short
quality contract. Public module timing covers96cases in opposite orders.
Inspect the paired review, when present, for whole-step performance; module
timings do not establish whole-step gains. Read TRITON_QWEN3.md
for source-specific evidence and QWEN3_TRITON_REPRODUCE.md for commands.

qwen3_triton implements the entire model's forward/backward computation.
The scripts compare native HF+Flash, complete Triton base and optimized,
including five ablations, multiupdate/full validation and module timings.
Training/data/FSDP/NCCL/AdamW are shared infrastructure, not operator gains.

No weights, dataset bytes, original experiment source directories, CUDA
caches, raw gradient archives, virtualenvs or profiler traces are included.
Obtain pinned official8B weights/data and run input verification separately.
SOURCE_MANIFEST.json records every copied file's hash. Third-party notices
are preserved under qwen3_triton/third_party; own source licensing is pending.

The common sharder retains its original optional vocabulary helpers for
source completeness; the public comparison entry fixes both options false.
Use the public scripts, not legacy training CLI names in helper docstrings.
'''
    if qualification['numerical_training_contract_qualified']:
        readme += '''
The bundled paired report covers all sixteen clean roles and passes the
frozen short-training/numerical/CUDA-coverage contract. It discloses any
interrupted-cohort recovery and excludes rejected resource-contaminated runs.
Numerical qualification uses the recorded tolerance contract; bitwise gradient
equivalence and training convergence are not claimed. Read the final report
for the measured comparison with HF and the five ablations.
'''
    if 'agent_data_runtime_acceptance' in qualification:
        readme += '\nThe latest data/runtime acceptance is recorded in results/qwen3_triton_v9_data_acceptance_v1.json.\n'
    (output/'README.md').write_text(readme)
    report=dict(complete=True,export_kind='local_source_candidate_not_released',
        model='Qwen/Qwen3-8B',published=False,own_source_license_selected=False,
        qualification=qualification,bitwise_gradient_equivalence_claimed=False,performance_adopted=False,
        files=rows,generated_readme_sha256=sha(output/'README.md'),
        scope=__doc__)
    (output/'SOURCE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,files=len(rows),source_bytes=sum(r['bytes'] for r in rows),output=str(output),published=False)),flush=True)

if __name__=='__main__':main()
