"""Check V10's explicit source delta and full 8B parameter contract on CPU/meta.

The historical V9 hash checker deliberately rejects modified kernels. This
check keeps that evidence immutable and verifies the new delta separately.
No BF16 numerical or GPU performance qualification is implied.
"""
from pathlib import Path
import argparse,ast,hashlib,json,subprocess
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from qwen3_triton.model import model_for_variant,HFOrderNorm
from qwen3_triton.kernels.gemm_tuning import load_plan

ROOT=Path(__file__).resolve().parents[1]
BASELINE='13dbdf984201a70ca4bba5fb38c72606cd33fbeb'


def digest(data):return hashlib.sha256(data).hexdigest()


def definition(source,path):
    node=ast.parse(source)
    for part in path.split('.'):
        node=next(n for n in node.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==part)
    return ast.dump(node,include_attributes=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/fsdp_8b.json')
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.resolve().is_relative_to(Path('/home/tzh'))
    assert not torch.cuda.is_initialized()
    manifest_path=ROOT/'results/qwen3_triton_source_port_v2.json'
    manifest=json.loads(manifest_path.read_text())
    changed=[];sources={}
    allowed={'qwen3_triton/kernels/linear.py','qwen3_triton/kernels/mlp_precision.py',
             'scripts/train_qwen3_triton_8b.py'}
    for row in manifest['files']:
        name=row['destination'];path=ROOT/name
        old=subprocess.check_output(['git','show',BASELINE+':'+name],cwd=ROOT)
        assert digest(old)==row['generated_sha256']
        current=path.read_bytes();sources[name]=digest(current);ast.parse(current)
        if current!=old:
            assert name in allowed,name
            changed.append(name)
            if name.startswith('scripts/'):
                definitions=('compare_first_gradients','evaluate','sha','local')
                loops=[]
                for source in (old,current):
                    loop=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
                    loops.append(ast.dump(loop,include_attributes=False))
                assert loops[0]==loops[1],'Training update and timing loop changed'
            else:
                definitions=('_gemm','_Linear') if name.endswith('/linear.py') else ('_gate_up','_GateUp.backward')
            for target in definitions:assert definition(current,target)==definition(old,target),(name,target)
    assert set(changed)==allowed
    init=manifest['generated_init'];assert digest((ROOT/init['path']).read_bytes())==init['sha256']
    plan_identity=load_plan(args.plan)
    protocol=json.loads(args.protocol.read_text())
    assert protocol['model_repo']=='Qwen/Qwen3-8B'
    config=Qwen3Config.from_pretrained(protocol['model'],local_files_only=True)
    assert (config.num_hidden_layers,config.hidden_size,config.intermediate_size,config.head_dim,config.vocab_size)==(36,4096,12288,128,151936)
    with torch.device('meta'):reference=Qwen3ForCausalLM(config)
    expected={n:tuple(p.shape) for n,p in reference.named_parameters()}
    assert len(expected)==399
    rows=[]
    for variant in ('triton_base','triton_optimized','ablate_attention','ablate_norm_rope','ablate_mlp','ablate_linear','ablate_loss'):
        with torch.device('meta'):model=model_for_variant(config,variant)
        assert all(p.is_meta for p in model.parameters())
        assert {n:tuple(p.shape) for n,p in model.named_parameters()}==expected
        assert sum(isinstance(m,HFOrderNorm) for m in model.modules())==145
        rows.append(dict(variant=variant,parameter_tensors=399,parameter_elements=sum(p.numel() for p in model.parameters())))
        del model
    assert not torch.cuda.is_initialized()
    for p in sorted((ROOT/'qwen3_triton').rglob('*.py')):
        sources[str(p.relative_to(ROOT))]=digest(p.read_bytes());ast.parse(p.read_text())
    result=dict(complete=True,passed=True,GPU_executed=False,CUDA_initialized=False,
        baseline_commit=BASELINE,changed_existing_model_files=changed,
        GEMM_kernel_AST_unchanged=True,gate_up_kernel_AST_unchanged=True,
        backward_and_autograd_AST_unchanged=True,variants=rows,plan=plan_identity,
        training_update_and_timing_loop_AST_unchanged=True,
        historical_V9_manifest_unchanged=True,sources=sources,scope=__doc__)
    result.update(checker_sha256=digest(Path(__file__).read_bytes()),
                  protocol_sha256=digest(args.protocol.read_bytes()))
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,passed=True,variants=len(rows),GPU_executed=False)))


if __name__=='__main__':main()
