"""Attribute the changed-input request failure without changing production math."""
import json
from pathlib import Path
import torch
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from qwen3_triton.inference import TritonQwen3Inference,kernels
from qwen3_triton.inference import model as im
from scripts import diagnose_qwen3_error_growth_v15 as harness
from scripts.verify_qwen3_inference_v15 import metrics,sha

@triton.jit
def _norm_probe(X,W,Y,P,H:tl.constexpr,WIDTH:tl.constexpr):
 lane=tl.arange(0,WIDTH);a=tl.full((WIDTH,),0.,tl.float32);b=a;c=a;d=a
 for start in range(H//(WIDTH*4)):
  off=start*WIDTH*4+lane*4
  x0=tl.load(X+off).to(tl.float32);x1=tl.load(X+off+1).to(tl.float32);x2=tl.load(X+off+2).to(tl.float32);x3=tl.load(X+off+3).to(tl.float32)
  a=a+x0*x0;b=b+x1*x1;c=c+x2*x2;d=d+x3*x3
 value=((a+b)+c)+d
 for step in tl.static_range(5 if WIDTH==32 else 9):value=value+tl.gather(value,lane^(WIDTH//2>>step),0)
 mean=tl.sum(tl.where(lane==0,value,0.),0)*(1./H);inv=tl.rsqrt(mean+1e-6)
 tl.store(P,mean);tl.store(P+1,inv)
 col=tl.arange(0,H);x=tl.load(X+col).to(tl.float32);w=tl.load(W+col).to(tl.float32)
 tl.store(Y+col,(x*inv).to(Y.dtype.element_ty).to(tl.float32)*w)

@torch.inference_mode()
def run(a,out,foreign):
 torch.set_num_threads(2);torch.manual_seed(42)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 root=Path(__file__).resolve().parents[1];files=[*sorted((root/'qwen3_triton').rglob('*.py')),Path(__file__),Path(harness.__file__)]
 hashes={str(p.relative_to(root)):sha(p) for p in files}
 model='/data1/tzh/models/Qwen/Qwen3-8B'
 c=TritonQwen3Inference.from_pretrained(model,reference_accumulation=True,prefill_tiling=True)
 h=AutoModelForCausalLM.from_pretrained(model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
 torch.randint(1,100000,(1,65),device='cuda');torch.randint(1,100000,(1,65),device='cuda');ids=torch.randint(1,100000,(1,65),device='cuda')
 refs={};handles=harness.attach_hf(h,refs);hc=DynamicCache();cc=c.new_cache(1,80);oc=c.new_cache(1,80);current=ids;phases=[];rows=[];norm_probes=[]
 try:
  for step in range(8):
   refs.clear();state={};start=cc.length
   y=h(current,past_key_values=hc,use_cache=True,logits_to_keep=1).logits
   x=c(current,past_key_values=cc,logits_to_keep=1).logits
   with harness.collect_candidate(c,state):z=c(current,past_key_values=oc,logits_to_keep=1).logits
   assert torch.equal(x,z),'Wrappers changed logits'
   for i in range(36):
    for aa,bb in [(cc.keys[i],oc.keys[i]),(cc.values[i],oc.values[i])]:assert torch.equal(aa[:,:,:cc.length],bb[:,:,:oc.length])
   hf_token=int(y[0,-1].argmax());triton_token=int(x[0,-1].argmax());top=y[0,-1].float().topk(2)
   phase=dict(step=step,start=start,HF_token=hf_token,Triton_token=triton_token,HF_top1_gap=float(top.values[0]-top.values[1]),**metrics(x,y));phases.append(phase);print(json.dumps(phase),flush=True)
   for i in range(36):
    p=f'layer.{i}.'
    for name in ['q','k']:
     normed=refs[p+name+'_normalized'].transpose(1,2);rotary,_=apply_rotary_pos_emb(normed,normed,refs['cos'],refs['sin']);refs[p+name+'_rope']=rotary.transpose(1,2).contiguous()
    refs[p+'attn_output']=refs[p+'o_proj_input'];refs[p+'activation']=refs[p+'down_proj_input']
   for tag,value in state.items():
    if not tag.endswith('_input'):rows.append(dict(step=step,tag=tag,kind='propagated',**harness.stats(value,refs[tag])))
   for i,layer in enumerate(c.model.layers):
    p=f'layer.{i}.'
    for tag,module,input_tag in [('input_norm',layer.input_layernorm,'input_norm_input'),('post_norm',layer.post_attention_layernorm,'attn_residual')]:
     inp=refs[p+input_tag].contiguous();actual=im.rms_norm(inp,module.weight,module.eps,reference_accumulation=True)
     info=dict(step=step,tag=p+tag,kind='local_norm_same_HF_input',**harness.stats(actual,refs[p+tag]));rows.append(info)
     if info['unequal'] and not norm_probes:
      for width in [32,512]:
       buf=torch.empty_like(inp);parts=torch.empty(2,device='cuda',dtype=torch.float32)
       _norm_probe[(1,)](inp,module.weight,buf,parts,4096,width,enable_fp_fusion=False)
       native_mean=inp.float().square().mean();native_inv=torch.rsqrt(native_mean+module.eps)
       norm_probes.append(dict(step=step,tag=p+tag,width=width,mean_inv=parts.cpu().tolist(),native_mean_inv=[float(native_mean),float(native_inv)],**harness.stats(buf,refs[p+tag])))
    for name in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
     module=getattr(layer.self_attn if name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp,name)
     actual=kernels.linear(refs[p+name+'_input'].contiguous(),module.weight,reference_accumulation=True,prefill_tiling=True)
     rows.append(dict(step=step,tag=p+name,kind='local_same_HF_input',**harness.stats(actual,refs[p+name])))
    actual=kernels.attention(refs[p+'q_rope'],hc.key_cache[i].contiguous(),hc.value_cache[i].contiguous(),start).view_as(refs[p+'attn_output'])
    rows.append(dict(step=step,tag=p+'attn_output',kind='local_HF_QKV_cache',**harness.stats(actual,refs[p+'attn_output'])))
   current=y[:,-1,:].argmax(-1,keepdim=True)
 finally:
  for handle in handles:handle.remove()
 for p in files:assert sha(p)==hashes[str(p.relative_to(root))],p
 result=dict(complete=True,diagnostic_only=True,seed=42,changed_input_case=1,input_tokens=ids.cpu().tolist(),phases=phases,rows=rows,norm_probes=norm_probes,source_sha256=hashes,foreign_context_observations=list(foreign),resource_clean=not foreign,scope='Exact changed-input benchmark prompt, teacher-forced HF continuation. Observed/unwrapped candidate logits and initialized cache are bitwise equal. No production math change.')
 raw=json.dumps(result,indent=2)+'\n';assert len(raw.encode())<=4*2**20;(out/'result.json').write_text(raw)
 print(json.dumps(dict(complete=True,rows=len(rows),norm_probes=norm_probes,resource_clean=not foreign)),flush=True)
 assert not any(row['unequal'] for row in rows if row['kind']=='local_norm_same_HF_input'),'Actual HF-input RMSNorm mismatch'
 assert all(p['HF_token']==p['Triton_token'] and p['passed'] for p in phases),'Changed-input continuation still differs or exceeds gates'

if __name__=='__main__':harness.run=run;harness.main()
