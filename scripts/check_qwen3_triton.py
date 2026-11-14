"""Check published source identity and actual Qwen3-8B architecture on CPU/meta.

This standalone check needs no original artifact sources. It checks generated
source hashes and every official parameter shape. Import-transformation AST
replay is a separate historical audit; this check does not replay it, load
weights, run CUDA, qualify BF16 numerics or time training.
"""
from pathlib import Path
import argparse,ast,hashlib,json
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
import qwen3_triton
from qwen3_triton.model import model_for_variant,HFOrderNorm

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/fsdp_8b.json')
    parser.add_argument('--source-manifest',type=Path,default=ROOT/'results/qwen3_triton_source_port_v2.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not torch.cuda.is_initialized()
    assert Path(qwen3_triton.__file__).resolve().is_relative_to(ROOT/'qwen3_triton')
    manifest=json.loads(args.source_manifest.read_text())
    for row in manifest['files']:
        destination=ROOT/row['destination']
        assert destination.resolve().is_relative_to(ROOT)
        assert sha(destination)==row['generated_sha256']
        ast.parse(destination.read_text())
    generated=manifest['generated_init'];assert sha(ROOT/generated['path'])==generated['sha256']
    protocol=json.loads(args.protocol.read_text());assert protocol['model_repo']=='Qwen/Qwen3-8B'
    config=Qwen3Config.from_pretrained(protocol['model'],local_files_only=True)
    assert config.num_hidden_layers==36 and config.hidden_size==4096 and config.head_dim==128
    assert config.vocab_size==151936 and config.num_attention_heads==32 and config.num_key_value_heads==8 and config.intermediate_size==12288
    with torch.device('meta'):reference=Qwen3ForCausalLM(config)
    expected={n:tuple(p.shape) for n,p in reference.named_parameters()};assert len(expected)==399
    rows=[]
    for variant in ('triton_base','triton_optimized','ablate_attention','ablate_norm_rope','ablate_mlp','ablate_linear','ablate_loss'):
        with torch.device('meta'):model=model_for_variant(config,variant)
        assert all(p.is_meta for p in model.parameters())
        assert {n:tuple(p.shape) for n,p in model.named_parameters()}==expected
        assert sum(isinstance(m,HFOrderNorm) for m in model.modules())==145
        expected_precision=manifest.get('attention_precision_options')
        if expected_precision is not None:
            assert all(layer.self_attn.precision_options==expected_precision for layer in model.model.layers)
        rows.append(dict(variant=variant,parameter_tensors=399,parameter_elements=sum(p.numel() for p in model.parameters()),HF_order_norm_modules=145,options=vars(model.options)))
        del model
    assert not torch.cuda.is_initialized()
    report=dict(complete=True,passed=True,model='Qwen/Qwen3-8B',layers=36,variants=rows,
        generated_source_identity_passed=True,source_transformation_replayed=False,GPU_executed=False,scope=__doc__,
        sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),args.source_manifest,args.protocol)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:handle.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,passed=True,model='Qwen/Qwen3-8B',variants=len(rows),GPU_executed=False)),flush=True)

if __name__=='__main__':main()
