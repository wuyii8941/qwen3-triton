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
    files += list(map(Path,('TRITON_QWEN3.md','requirements-qwen3-triton.txt')))
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
    readme='''# Qwen3-8B Triton 源码导出

此目录是用于审阅的源码包，包含模型计算、数据读取与验证辅助工具。
模型权重、语料、虚拟环境、缓存、原始梯度和 profiler trace 不包含在内。

完整训练、推理及验证范围请查看 [训练指南](https://github.com/wuyii8941/qwen3-triton/blob/main/TRAINING.md)
和 [推理指南](https://github.com/wuyii8941/qwen3-triton/blob/main/INFERENCE.md)。
运行前需要准备本地官方权重和已校验的数据。

`SOURCE_MANIFEST.json` 记录导出文件的哈希。第三方许可保存在
`qwen3_triton/third_party/`；项目自有源码许可证尚未选定。
'''
    if qualification['numerical_training_contract_qualified']:
        readme += '\n包内包含短程数值和 GPU 覆盖审计摘要；这些检查不证明长期训练收敛。\n'
    if 'agent_data_runtime_acceptance' in qualification:
        readme += '\n包内包含数据与运行环境检查摘要。\n'
    (output/'README.md').write_text(readme)
    report=dict(complete=True,export_kind='local_source_candidate_not_released',
        model='Qwen/Qwen3-8B',published=False,own_source_license_selected=False,
        qualification=qualification,bitwise_gradient_equivalence_claimed=False,performance_adopted=False,
        files=rows,generated_readme_sha256=sha(output/'README.md'),
        scope=__doc__)
    (output/'SOURCE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,files=len(rows),source_bytes=sum(r['bytes'] for r in rows),output=str(output),published=False)),flush=True)

if __name__=='__main__':main()
