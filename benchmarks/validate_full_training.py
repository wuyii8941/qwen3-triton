"""Same-weight full-model loss and EVERY parameter-gradient comparison.

CPU snapshots avoid keeping two full GPU models/gradient sets resident.
No optimizer updates; BF16 tolerances are reported per parameter, not hidden
behind a mean. This is numerical parity, not convergence validation.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
from training.hf_operators import training_operators
from training.fused_loss import fused_linear_cross_entropy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seq',type=int,default=128)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',default='repeat,rms,swiglu,qk,operators,fused_loss,compile')
    p.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16')
    p.add_argument('--layer-diagnostics',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.layer_diagnostics and 'compile' in a.variants.split(','):
        raise ValueError('Hooks would alter compile graphs; diagnose compile separately without hooks')
    torch.set_num_threads(4)
    torch.manual_seed(20261102)
    path='/data1/tzh/models/Qwen/Qwen3-1.7B'
    m=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=getattr(torch,a.dtype),
        attn_implementation='sdpa').cuda().train()
    t=AutoTokenizer.from_pretrained(path,local_files_only=True)
    corpus=json.loads(Path('artifacts/frequency_wikitext_data.json').read_text())
    ids=torch.tensor(t(corpus['splits']['validation']['text'][:10000],add_special_tokens=False)['input_ids'][:a.seq],device='cuda')[None]
    report={'scope':'Complete 28-layer same-weight gradient comparison, no optimizer, no checkpointing',
            'dtype':a.dtype,'backend':'math' if a.dtype=='float32' else 'flash',
            'seq':a.seq,'gates':{'loss_abs':.01,'parameter_gradient_relative_l2_max':.05},'candidates':{}}
    captures={};handles=[]
    if a.layer_diagnostics:
        def hook_for(index):
            def hook(module,inputs,output):
                tensor=output[0] if isinstance(output,tuple) else output
                captures.setdefault(index,{})['output']=tensor.detach().cpu().clone()
                def grad_hook(grad):
                    captures.setdefault(index,{})['grad_output']=grad.detach().cpu().clone()
                tensor.register_hook(grad_hook)
            return hook
        handles=[layer.register_forward_hook(hook_for(i)) for i,layer in enumerate(m.model.layers)]
    backend=SDPBackend.MATH if a.dtype=='float32' else SDPBackend.FLASH_ATTENTION
    with sdpa_kernel(backend):
        loss=m(ids,labels=ids,use_cache=False).loss
        loss.backward();base_loss=float(loss.detach())
        gradients={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters()}
        baseline_captures=captures.copy();captures.clear()
        for variant in a.variants.split(','):
            m.zero_grad(set_to_none=True)
            def loss_fn(tokens):
                if variant=='fused_loss':
                    h=m.model(tokens,use_cache=False).last_hidden_state
                    return fused_linear_cross_entropy(h[:,:-1].reshape(-1,h.shape[-1]),m.lm_head.weight,tokens[:,1:].reshape(-1))
                return m(tokens,labels=tokens,use_cache=False).loss
            fn=torch.compile(loss_fn) if variant=='compile' else loss_fn
            with training_operators(m,rms=variant in ('rms','operators','rms_hf','rounded'),
                    swiglu=variant in ('swiglu','operators','swiglu_hf','swiglu_hf_bwd','rounded'),
                    qk=variant in ('qk','operators'),hf_semantics=('triton_backward' if variant=='swiglu_hf_bwd'
                        else variant in ('rms_hf','swiglu_hf','rounded'))):
                actual=fn(ids);actual.backward()
            errors={}
            for n,param in m.named_parameters():
                if param.grad is None:raise AssertionError(f'Missing gradient: {n}')
                g=param.grad.detach().cpu().float();ref=gradients[n].float()
                errors[n]=dict(relative_l2=float((g-ref).norm()/ref.norm().clamp_min(1e-12)),
                    max_abs=float((g-ref).abs().max()),reference_norm=float(ref.norm()),
                    difference_norm=float((g-ref).norm()),finite=bool(torch.isfinite(g).all()))
            delta=abs(float(actual.detach())-base_loss)
            report['candidates'][variant]=dict(loss=float(actual.detach()),reference_loss=base_loss,
                loss_abs_error=delta,gradients=errors,
                accepted=delta<.01 and all(r['finite'] and r['relative_l2']<.05 for r in errors.values()))
            if a.layer_diagnostics:
                layer_errors={}
                for i,values in captures.items():
                    layer_errors[i]={}
                    for key,value in values.items():
                        ref=baseline_captures[i][key].float();g=value.float()
                        layer_errors[i][key]=dict(relative_l2=float((g-ref).norm()/ref.norm().clamp_min(1e-12)),
                            max_abs=float((g-ref).abs().max()))
                report['candidates'][variant]['layers']=layer_errors
                captures.clear()
            a.output.write_text(json.dumps(report,indent=2))
            print(variant,report['candidates'][variant]['accepted'],max(r['relative_l2'] for r in errors.values()),flush=True)
    for handle in handles:handle.remove()


if __name__=='__main__':main()
