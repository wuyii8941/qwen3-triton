"""Isolated forward/backward module timings on actual pretrained Qwen3-8B data.

This is not an end-to-end training measurement: no FSDP/NCCL/optimizer. Inputs
come from a verified complete36layer8B forward, weights from the same pinned
checkpoint. All timed candidates receive an untimed no-fallback VJP first.
"""
from pathlib import Path
import argparse,copy,gc,json,statistics
import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Qwen3Config,AutoModelForCausalLM
from training.data import Blocks
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention,Qwen3MLP,Qwen3RMSNorm,apply_rotary_pos_emb
from torch.nn.attention import sdpa_kernel,SDPBackend
from qwen3_triton._model import TritonLinear
from qwen3_triton._model_optimized import options_for_variant
from qwen3_triton.kernels.rotary import apply_rotary
from qwen3_triton.kernels.loss import causal_cross_entropy
from qwen3_triton.audit import NoFallback
from training.input_manifest import validate
from qwen3_triton.model import model_for_variant
from qwen3_triton.kernels.math_precision import rms_norm,norm_rope
from qwen3_triton.gradient_archive import sha
ROOT=Path(__file__).resolve().parents[1];OUT=None

class NormPair(nn.Module):
    def __init__(self,weights,eps,variant,positions):
        super().__init__();self.q=nn.Parameter(weights[0]);self.k=nn.Parameter(weights[1]);self.eps=eps;self.variant=variant;self.positions=positions
    def forward(self,q,k):
        cos,sin=self.positions
        if self.variant=='hf':
            # Reference modules use native HF RMSNorm and rotary functions.
            q=self.q_native(q).transpose(1,2);k=self.k_native(k).transpose(1,2)
            q,k=apply_rotary_pos_emb(q,k,cos,sin)
            return q.transpose(1,2).contiguous(),k.transpose(1,2).contiguous()
        if self.variant=='triton_optimized':return norm_rope(q,self.q,cos,sin,self.eps),norm_rope(k,self.k,cos,sin,self.eps)
        return apply_rotary(rms_norm(q,self.q,self.eps),cos,sin),apply_rotary(rms_norm(k,self.k,self.eps),cos,sin)

class Loss(nn.Module):
    def __init__(self,variant,ids):super().__init__();self.variant=variant;self.ids=ids
    def forward(self,x):
        if self.variant=='hf':return F.cross_entropy(x[:,:-1].contiguous().float().view(-1,x.shape[-1]),self.ids[:,1:].contiguous().view(-1))
        return causal_cross_entropy(x,self.ids,optimized=self.variant=='triton_optimized')

def measure(module,forward,templates,upstreams,variant):
    def once(audit=False,events=False):
        module.zero_grad(set_to_none=True);leaves=[t.detach().requires_grad_() for t in templates]
        stamps=[torch.cuda.Event(enable_timing=True) for _ in range(3)] if events else None
        if events:stamps[0].record()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            if audit and variant!='hf':
                with NoFallback():outputs=forward(module,leaves)
                if events:stamps[1].record()
                with NoFallback(backward=True):torch.autograd.backward(outputs,upstreams)
            else:
                outputs=forward(module,leaves)
                if events:stamps[1].record()
                torch.autograd.backward(outputs,upstreams)
        if events:stamps[2].record()
        if audit:
            assert all(torch.isfinite(t).all() for t in outputs)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
            assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in leaves)
        return stamps
    once(audit=True)
    for _ in range(5):once()
    module.zero_grad(set_to_none=True);torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    stamps=[once(events=True) for _ in range(20)];torch.cuda.synchronize()
    forward_ms=[a.elapsed_time(b) for a,b,c in stamps];backward_ms=[b.elapsed_time(c) for a,b,c in stamps]
    return dict(forward_ms=forward_ms,backward_ms=backward_ms,median_forward_ms=statistics.median(forward_ms),median_backward_ms=statistics.median(backward_ms),peak_extra_allocated_gib=(torch.cuda.max_memory_allocated()-baseline)/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,untimed_no_fallback_VJP_finite=True)


@torch.no_grad()
def capture_reference(protocol,identity):
    """Capture real module inputs from a complete36layer pretrained8B forward."""
    model=AutoModelForCausalLM.from_pretrained(protocol['model'],local_files_only=True,
        torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
    assert len(model.model.layers)==36 and len(dict(model.named_parameters()))==399
    assert model.config.hidden_size==4096 and model.config.head_dim==128 and model.config.vocab_size==151936
    ids_cpu=Blocks(Path(protocol['corpus'])/'train.bin',512,training=True,seed=protocol['seed'])[0].unsqueeze(0).contiguous()
    captured={};handles=[]
    names=['model.embed_tokens','lm_head']
    for layer in (0,17,35):
        prefix=f'model.layers.{layer}.'
        names.extend(prefix+suffix for suffix in ('self_attn','self_attn.q_norm','self_attn.k_norm','mlp'))
    for name in names:
        def pre(module,args,kwargs,name=name):
            value=args[0] if args else kwargs['hidden_states']
            entry=dict(input=value.detach().cpu())
            if name.endswith('self_attn'):
                assert kwargs.get('attention_mask') is None
                entry['positions']=tuple(t.detach().cpu() for t in kwargs['position_embeddings'])
            captured[name]=entry
        handles.append(model.get_submodule(name).register_forward_pre_hook(pre,with_kwargs=True))
    try:
        ids=ids_cpu.cuda()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):out=model(ids,labels=ids,use_cache=False)
        assert torch.isfinite(out.logits).all() and torch.isfinite(out.loss)
        refs=dict(logits=out.logits.detach().cpu(),loss=out.loss.detach().cpu())
        assert refs['logits'].shape==(1,512,151936) and refs['logits'].dtype==torch.bfloat16
        assert torch.equal(captured['model.embed_tokens']['input'],ids_cpu)
    finally:
        for handle in handles:handle.remove()
    path=OUT/'reference_capture.pt';torch.save(dict(reference=captured,refs=refs,ids=ids_cpu),path)
    proof=dict(complete=True,model='Qwen/Qwen3-8B',layers=36,parameter_tensors=399,
        input_manifest_sha256=identity,capture_sha256=sha(path),captured_modules=names,
        scope='Fresh complete36layer8B nativeHF+Flash forward, first frozen training input. Captures supply isolated module timings; no end-to-end speed claim.')
    (OUT/'reference_capture.json').write_text(json.dumps(proof,indent=2)+'\n')
    del out,model,ids;gc.collect();torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated()<1024**3
    return captured,refs,proof

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,default=ROOT/'configs/fsdp_8b.json');parser.add_argument('--input-manifest',type=Path,default=ROOT/'artifacts/official_qwen3_inputs_v1.json');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();OUT=args.output
    if OUT.resolve().is_relative_to(Path('/home/tzh')):parser.error('Output under /home/tzh is forbidden')
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(20261114)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.cuda.set_per_process_memory_fraction(41.875*2**30/torch.cuda.get_device_properties().total_memory)
    protocol=json.loads(args.protocol.read_text())
    assert protocol['model_repo']=='Qwen/Qwen3-8B'
    identity=validate(args.input_manifest,args.protocol,protocol)
    capture,refs,capture_proof=capture_reference(protocol,identity)
    model_dir=Path(protocol['model']);config=Qwen3Config.from_pretrained(model_dir,local_files_only=True);config._attn_implementation='sdpa'
    assert config.num_hidden_layers==36 and config.hidden_size==4096 and config.head_dim==128
    index=json.loads((model_dir/'model.safetensors.index.json').read_text())['weight_map']
    def state(prefix,names):
        result={}
        for name in names:
            key=prefix+name
            with safe_open(model_dir/index[key],framework='pt',device='cpu') as handle:result[name]=handle.get_tensor(key)
        assert all(t.dtype==torch.bfloat16 for t in result.values());return result
    rows=[]
    groups=[(layer,kind) for layer in (0,17,35) for kind in ('attention','norm_rope','mlp')]+[(None,'lm_head'),(None,'loss')]
    for layer,kind in groups:
        prefix=f'model.layers.{layer}.' if layer is not None else ''
        if kind=='attention':
            variants=['hf','triton_base','triton_optimized','ablate_attention','ablate_norm_rope','ablate_linear'];positions=tuple(t.cuda() for t in capture[prefix+'self_attn']['positions'])
            templates=[capture[prefix+'self_attn']['input'].cuda()];shape=templates[0].shape
            weights=state(prefix+'self_attn.',[n+'.weight' for n in ('q_proj','k_proj','v_proj','o_proj','q_norm','k_norm')])
        elif kind=='norm_rope':
            variants=['hf','triton_base','triton_optimized'];positions=tuple(t.cuda() for t in capture[prefix+'self_attn']['positions'])
            templates=[capture[prefix+'self_attn.'+name]['input'].cuda() for name in ('q_norm','k_norm')];shape=None
            weights=state(prefix+'self_attn.',['q_norm.weight','k_norm.weight'])
        elif kind=='mlp':
            variants=['hf','triton_base','triton_optimized','ablate_mlp','ablate_linear'];templates=[capture[prefix+'mlp']['input'].cuda()];shape=templates[0].shape
            weights=state(prefix+'mlp.',[n+'.weight' for n in ('gate_proj','up_proj','down_proj')])
        elif kind=='lm_head':
            variants=['hf','triton_base','triton_optimized'];templates=[capture['lm_head']['input'].cuda()];shape=refs['logits'].shape;weights=state('lm_head.',['weight'])
        else:
            variants=['hf','triton_base','triton_optimized'];templates=[refs['logits'].cuda()];shape=();weights={}
        generator=torch.Generator().manual_seed(20261114+(layer or 0)*10+('attention','norm_rope','mlp','lm_head','loss').index(kind))
        if kind=='norm_rope':upstreams=[torch.randn(t.shape,generator=generator).to(torch.bfloat16).cuda() for t in templates]
        else:upstreams=[torch.randn(shape,generator=generator).to(torch.float32 if kind=='loss' else torch.bfloat16).cuda()]
        for order in (1,2):
            for variant in variants if order==1 else reversed(variants):
                if kind in ('attention','mlp'):
                    with torch.device('meta'):
                        if variant=='hf':module=Qwen3Attention(copy.deepcopy(config),layer_idx=layer) if kind=='attention' else Qwen3MLP(copy.deepcopy(config))
                        else:
                            whole=model_for_variant(copy.deepcopy(config),variant);module=whole.model.layers[layer].self_attn if kind=='attention' else whole.model.layers[layer].mlp;del whole
                    module.load_state_dict(weights,strict=True,assign=True);module=module.cuda().train()
                    if kind=='attention':
                        forward=(lambda module,x:(module(x[0],position_embeddings=positions,attention_mask=None)[0],)) if variant=='hf' else (lambda module,x:(module(x[0],positions),))
                    else:forward=lambda module,x:(module(x[0]),)
                elif kind=='norm_rope':
                    module=NormPair([weights['q_norm.weight'],weights['k_norm.weight']],config.rms_norm_eps,variant,positions)
                    if variant=='hf':
                        module.q_native=Qwen3RMSNorm(128,eps=config.rms_norm_eps);module.k_native=Qwen3RMSNorm(128,eps=config.rms_norm_eps)
                        module.q_native.weight=module.q;module.k_native.weight=module.k
                        del module.q,module.k
                    module=module.cuda().train();forward=lambda module,x:module(*x)
                elif kind=='lm_head':
                    with torch.device('meta'):module=nn.Linear(4096,151936,bias=False) if variant=='hf' else TritonLinear(4096,151936,config.initializer_range,options_for_variant(variant))
                    module.load_state_dict(weights,strict=True,assign=True);module=module.cuda().train();forward=lambda module,x:(module(x[0]),)
                else:
                    ids=capture['model.embed_tokens']['input'].cuda();module=Loss(variant,ids);forward=lambda module,x:(module(x[0]),)
                values=measure(module,forward,templates,upstreams,variant)
                row=dict(layer=layer,module=kind,variant=variant,order=order,input_shapes=[list(t.shape) for t in templates],parameter_count=sum(p.numel() for p in module.parameters()),**values);rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('forward_ms','backward_ms')}),flush=True)
                (OUT/'progress.json').write_text(json.dumps(dict(complete=False,rows=rows),indent=2)+'\n');del module,forward;gc.collect();torch.cuda.empty_cache()
        del templates,upstreams,weights
    files=[Path(__file__),args.protocol,args.input_manifest,OUT/'reference_capture.json',*sorted((ROOT/'qwen3_triton').rglob('*.py'))]
    result=dict(complete=True,model='Qwen/Qwen3-8B',input_manifest_sha256=identity,rows=rows,reference_capture_sha256=capture_proof['capture_sha256'],math_policy=dict(compute='BF16',TF32=False,allow_bf16_reduced_precision_reduction=False,deterministic_fill=True),scope=__doc__,sources={str(p):sha(p) for p in files})
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
