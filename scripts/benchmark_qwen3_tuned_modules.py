"""Integrated module VJPs and paired timing: HF, frozen V9, and opt-in GEMM plan.

Real inputs are reused from the hashed accepted full-8B capture. These local
BF16 parameter gradients exclude FSDP/FP32 master gradients and optimizer.
Passing this screen does not qualify the complete training numerical contract.
"""
from pathlib import Path
from contextlib import nullcontext
import argparse, copy, gc, json
import torch
from torch import nn
from safetensors import safe_open
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention,Qwen3MLP
from torch.nn.attention import sdpa_kernel,SDPBackend
from qwen3_triton.model import model_for_variant
from qwen3_triton._model import TritonLinear
from qwen3_triton._model_optimized import options_for_variant
from qwen3_triton.kernels.gemm_tuning import load_plan,enabled
from qwen3_triton.audit import NoFallback
from qwen3_triton.gradient_archive import sha
from benchmark_qwen3_triton_modules import measure


def compare(actual,reference):
    assert actual.shape==reference.shape and actual.dtype==reference.dtype
    numerator=denominator=maximum=0.
    a,b=actual.reshape(-1),reference.reshape(-1)
    finite=True
    for start in range(0,a.numel(),2**20):
        x,y=a[start:start+2**20].float(),b[start:start+2**20].float()
        delta=x-y
        numerator+=float(delta.square().sum());denominator+=float(y.square().sum())
        maximum=max(maximum,float(delta.abs().max()))
        finite &= bool(torch.isfinite(x).all())
    absolute=numerator**.5;norm=denominator**.5
    threshold=.001*max(norm,1.e-8)+1.e-6
    return dict(finite=finite,absolute_error_norm=absolute,reference_norm=norm,
                relative_l2=absolute/max(norm,1.e-8),max_abs=maximum,
                threshold=threshold,passed=finite and absolute<threshold)


def snapshot(module,forward,templates,upstream,variant):
    module.zero_grad(set_to_none=True)
    leaves=[x.detach().requires_grad_() for x in templates]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with NoFallback() if variant!='hf' else nullcontext():outputs=forward(module,leaves)
        with NoFallback(backward=True) if variant!='hf' else nullcontext():
            torch.autograd.backward(outputs,upstream)
    result={f'output_{i}':x.detach().cpu() for i,x in enumerate(outputs)}
    result.update({f'input_gradient_{i}':x.grad.detach().cpu() for i,x in enumerate(leaves)})
    result.update({'parameter_gradient_'+name:p.grad.detach().cpu() for name,p in module.named_parameters()})
    assert all(torch.isfinite(t).all() for t in result.values())
    module.zero_grad(set_to_none=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('/data1/tzh/models/Qwen/Qwen3-8B'))
    args=parser.parse_args();out=args.output.resolve()
    assert not out.is_relative_to(Path('/home/tzh'))
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(20261114)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    prop=torch.cuda.get_device_properties(0)
    assert (prop.major,prop.minor)==(8,6)
    torch.cuda.set_per_process_memory_fraction(41.875*2**30/prop.total_memory)
    plan_identity=load_plan(args.plan)
    proof=json.loads(args.capture.with_suffix('.json').read_text())
    assert proof['complete'] and proof['layers']==36 and sha(args.capture)==proof['capture_sha256']
    capture=torch.load(args.capture,map_location='cpu',weights_only=True)['reference']
    config=Qwen3Config.from_pretrained(args.model,local_files_only=True)
    config._attn_implementation='sdpa'
    assert config.num_hidden_layers==36 and config.hidden_size==4096 and config.head_dim==128
    index=json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    rows=[]
    for layer,kind in [(layer,kind) for layer in (0,17,35) for kind in ('attention','mlp')]+[(None,'lm_head')]:
        prefix=f'model.layers.{layer}.' if layer is not None else ''
        if kind=='attention':
            prefix+='self_attn.'
            names=[name+'.weight' for name in ('q_proj','k_proj','v_proj','o_proj','q_norm','k_norm')]
            template=capture[prefix[:-1]]['input'].cuda()
            positions=tuple(t.cuda() for t in capture[prefix[:-1]]['positions'])
            shape=template.shape
        elif kind=='mlp':
            prefix+='mlp.'
            names=[name+'.weight' for name in ('gate_proj','up_proj','down_proj')]
            template=capture[prefix[:-1]]['input'].cuda();shape=template.shape
        else:
            prefix='lm_head.';names=['weight'];template=capture['lm_head']['input'].cuda()
            shape=(1,512,151936)
        weights={}
        for name in names:
            key=prefix+name
            with safe_open(args.model/index[key],framework='pt',device='cpu') as handle:weights[name]=handle.get_tensor(key)
        generator=torch.Generator().manual_seed(20261114+(layer or 0)*10+('attention','mlp','lm_head').index(kind))
        upstream=[torch.randn(shape,generator=generator).to(torch.bfloat16).cuda()]
        references={}
        # First-order VJPs precede all timing; retain no dense gradients on disk.
        for variant in ('hf','v9','v10'):
            with torch.device('meta'):
                if kind=='lm_head':
                    module=nn.Linear(4096,151936,bias=False) if variant=='hf' else TritonLinear(4096,151936,config.initializer_range,options_for_variant('triton_optimized'))
                elif variant=='hf':
                    module=Qwen3Attention(copy.deepcopy(config),layer_idx=layer) if kind=='attention' else Qwen3MLP(copy.deepcopy(config))
                else:
                    whole=model_for_variant(copy.deepcopy(config),'triton_optimized')
                    module=whole.model.layers[layer].self_attn if kind=='attention' else whole.model.layers[layer].mlp
                    del whole
            module.load_state_dict(weights,strict=True,assign=True);module=module.cuda().train()
            if kind=='attention':
                forward=(lambda module,x:(module(x[0],position_embeddings=positions,attention_mask=None)[0],)) if variant=='hf' else (lambda module,x:(module(x[0],positions),))
            else:forward=lambda module,x:(module(x[0]),)
            with enabled(variant=='v10'):
                refs=snapshot(module,forward,[template],upstream,variant)
            if variant in ('hf','v9'):references[variant]=refs
            else:
                assert refs.keys()==references['v9'].keys()==references['hf'].keys()
                checks={key:compare(refs[key],references['v9'][key]) for key in refs}
                hf_checks={key:compare(refs[key],references['hf'][key]) for key in refs}
                assert all(check['passed'] for check in checks.values()),(layer,kind,checks)
                rows.append(dict(layer=layer,module=kind,variant=variant,comparison='v10_vs_v9',numeric=checks,
                    hf_numeric_descriptive=hf_checks,local_VJP_passed=True))
                del refs
            del module,forward;gc.collect();torch.cuda.empty_cache()
        del references
        for order in (1,2):
            for variant in (('hf','v9','v10') if order==1 else ('v10','v9','hf')):
                with torch.device('meta'):
                    if kind=='lm_head':module=nn.Linear(4096,151936,bias=False) if variant=='hf' else TritonLinear(4096,151936,config.initializer_range,options_for_variant('triton_optimized'))
                    elif variant=='hf':module=Qwen3Attention(copy.deepcopy(config),layer_idx=layer) if kind=='attention' else Qwen3MLP(copy.deepcopy(config))
                    else:
                        whole=model_for_variant(copy.deepcopy(config),'triton_optimized')
                        module=whole.model.layers[layer].self_attn if kind=='attention' else whole.model.layers[layer].mlp;del whole
                module.load_state_dict(weights,strict=True,assign=True);module=module.cuda().train()
                if kind=='attention':
                    forward=(lambda module,x:(module(x[0],position_embeddings=positions,attention_mask=None)[0],)) if variant=='hf' else (lambda module,x:(module(x[0],positions),))
                else:forward=lambda module,x:(module(x[0]),)
                with enabled(variant=='v10'):values=measure(module,forward,[template],upstream,variant)
                row=dict(layer=layer,module=kind,variant=variant,order=order,**values)
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('forward_ms','backward_ms')}),flush=True)
                (out/'progress.json').write_text(json.dumps(dict(complete=False,rows=rows,plan=plan_identity),indent=2)+'\n')
                del module,forward;gc.collect();torch.cuda.empty_cache()
        del weights,template,upstream;gc.collect();torch.cuda.empty_cache()
    assert len(rows)==49
    root=Path(__file__).resolve().parents[1]
    files=[Path(__file__),root/'scripts/benchmark_qwen3_triton_modules.py',*sorted((root/'qwen3_triton').rglob('*.py'))]
    result=dict(complete=True,rows=rows,plan=plan_identity,capture_sha256=proof['capture_sha256'],
        sources={str(p.relative_to(root)):sha(p) for p in files},
        full_training_numerical_contract_qualified=False,scope=__doc__)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
