"""Reference-only boundary diagnostics for the original short/decode failures.

Runs the unmodified public inference API. Wrappers retain GPU intermediate
references only during each phase and must reproduce the unwrapped logits and
initialized KV contents bitwise. No tensors/checkpoints are exported.
"""
import argparse,contextlib,hashlib,json,os,subprocess,threading
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from qwen3_triton.inference import TritonQwen3Inference,kernels
from qwen3_triton.inference import model as im
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from scripts.verify_qwen3_inference_v15 import metrics,sha

ROOT=Path(__file__).resolve().parents[1]

def stats(x,y):
 assert x.shape==y.shape,(x.shape,y.shape)
 d=x.float()-y.float()
 return dict(unequal=int((x!=y).sum()),numel=x.numel(),max_abs=float(d.abs().max()),mean_abs=float(d.abs().mean()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(y.float()).clamp_min(1e-12)),mean_offset=float(d.mean()))

def attach_hf(h,refs):
 handles=[]
 def output(tag,input_tag=None):
  def hook(module,inp,out):
   refs[tag]=out[0] if isinstance(out,tuple) else out
   if input_tag is not None:refs[input_tag]=inp[0]
  return hook
 handles.append(h.model.embed_tokens.register_forward_hook(output('embedding')))
 handles.append(h.model.rotary_emb.register_forward_hook(lambda module,inp,out:refs.update(cos=out[0],sin=out[1])))
 for i,layer in enumerate(h.model.layers):
  prefix=f'layer.{i}.'
  handles.append(layer.input_layernorm.register_forward_hook(output(prefix+'input_norm',prefix+'input_norm_input')))
  handles.append(layer.post_attention_layernorm.register_forward_hook(output(prefix+'post_norm',prefix+'attn_residual')))
  handles.append(layer.register_forward_hook(output(prefix+'output')))
  for name in ['q_proj','k_proj','v_proj','o_proj']:
   tag=prefix+name
   handles.append(getattr(layer.self_attn,name).register_forward_hook(output(tag,tag+'_input')))
  for name in ['gate_proj','up_proj','down_proj']:
   tag=prefix+name
   handles.append(getattr(layer.mlp,name).register_forward_hook(output(tag,tag+'_input')))
  handles.append(layer.self_attn.q_norm.register_forward_hook(output(prefix+'q_normalized')))
  handles.append(layer.self_attn.k_norm.register_forward_hook(output(prefix+'k_normalized')))
 handles.append(h.model.norm.register_forward_hook(output('final_norm','final_norm_input')))
 handles.append(h.lm_head.register_forward_hook(output('logits','logits_input')))
 return handles

@contextlib.contextmanager
def collect_candidate(c,state):
 linear_names={};norm_names={};rope_names={};gate_names={}
 for i,layer in enumerate(c.model.layers):
  p=f'layer.{i}.'
  for name in ['q_proj','k_proj','v_proj','o_proj']:linear_names[id(getattr(layer.self_attn,name).weight)]=p+name
  for name in ['gate_proj','up_proj','down_proj']:linear_names[id(getattr(layer.mlp,name).weight)]=p+name
  norm_names[id(layer.input_layernorm.weight)]=p+'input_norm';norm_names[id(layer.post_attention_layernorm.weight)]=p+'post_norm'
  rope_names[id(layer.self_attn.q_norm.weight)]=p+'q_rope';rope_names[id(layer.self_attn.k_norm.weight)]=p+'k_rope'
  gate_names[id(layer.mlp.gate_proj.weight)]=p+'activation'
 norm_names[id(c.model.norm.weight)]='final_norm';linear_names[id(c.lm_head.weight)]='logits'
 original={name:getattr(im,name) for name in ['embedding','rms_norm','norm_rope','residual_add','swiglu','gate_up_swiglu']}
 original_linear=kernels.linear;original_attention=kernels.attention
 counters=dict(residual=0,attention=0,activation=0)
 def save(tag,y):state[tag]=y;return y
 def embedding(*args,**kw):return save('embedding',original['embedding'](*args,**kw))
 def norm(x,w,*args,**kw):return save(norm_names[id(w)],original['rms_norm'](x,w,*args,**kw))
 def rope(x,w,*args,**kw):return save(rope_names[id(w)],original['norm_rope'](x,w,*args,**kw))
 def linear(x,w,*args,**kw):
  tag=linear_names[id(w)];state[tag+'_input']=x
  return save(tag,original_linear(x,w,*args,**kw))
 def residual(*args,**kw):
  n=counters['residual'];counters['residual']+=1
  return save(f'layer.{n//2}.'+('attn_residual' if n%2==0 else 'output'),original['residual_add'](*args,**kw))
 def attention(*args,**kw):
  n=counters['attention'];counters['attention']+=1;y=original_attention(*args,**kw)
  state[f'layer.{n}.attn_output']=y.view(*y.shape[:2],-1)
  return y
 def swiglu(*args,**kw):
  n=counters['activation'];counters['activation']+=1
  return save(f'layer.{n}.activation',original['swiglu'](*args,**kw))
 def fused(x,wg,wu,*args,**kw):return save(gate_names[id(wg)],original['gate_up_swiglu'](x,wg,wu,*args,**kw))
 im.embedding=embedding;im.rms_norm=norm;im.norm_rope=rope;im.residual_add=residual;im.swiglu=swiglu;im.gate_up_swiglu=fused
 kernels.linear=linear;kernels.attention=attention
 try:yield
 finally:
  for name,fn in original.items():setattr(im,name,fn)
  kernels.linear=original_linear;kernels.attention=original_attention

@torch.inference_mode()
def run(a,out,foreign):
 torch.set_num_threads(2);torch.manual_seed(20261116)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 files=[*sorted((ROOT/'qwen3_triton').rglob('*.py')),Path(__file__),ROOT/'scripts/verify_qwen3_inference_v15.py']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
 model_path='/data1/tzh/models/Qwen/Qwen3-8B'
 c=TritonQwen3Inference.from_pretrained(model_path)
 h=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
 expected=json.loads((ROOT/'results/qwen3_triton_inference_v15_prefill_api_HF_gates_v1.json').read_text())
 refs={};handles=attach_hf(h,refs);rows=[];phases=[]
 try:
  for length in [17,65]:
   ids=torch.randint(1,100000,(1,length+4),device='cuda');hc=DynamicCache();baseline=c.new_cache(1,length+8);observed=c.new_cache(1,length+8)
   for step in range(5):
    refs.clear();state={};current=ids[:,:length].contiguous() if step==0 else ids[:,length+step-1:length+step].contiguous()
    start=baseline.length
    y=h(current,past_key_values=hc,use_cache=True,logits_to_keep=0).logits
    x=c(current,past_key_values=baseline,logits_to_keep=0).logits
    with collect_candidate(c,state):z=c(current,past_key_values=observed,logits_to_keep=0).logits
    assert torch.equal(x,z),'Observation wrappers changed logits'
    for i in range(36):
     for aa,bb in [(baseline.keys[i],observed.keys[i]),(baseline.values[i],observed.values[i])]:assert torch.equal(aa[:,:,:baseline.length],bb[:,:,:observed.length]),'Observation wrappers changed KV'
    identity=dict(prompt_length=length,step=step,start=start,input_ids=current.cpu().tolist())
    values=metrics(x,y)
    old=next(row for row in expected['cases'] if row['batch']==1 and row['kind']==('prefill' if step==0 else 'cached_decode') and (row.get('length')==length if step==0 else row.get('after_tokens')==start))
    assert all(values[k]==old[k] for k in values),'Original numerical case not reproduced'
    phase=dict(**identity,metrics=values,observation_logits_bitwise_equal=True,initialized_KV_bitwise_equal=True)
    phases.append(phase);print(json.dumps(phase),flush=True)
    for i in range(36):
     p=f'layer.{i}.'
     for name in ['q','k']:
      normed=refs[p+name+'_normalized'].transpose(1,2)
      rotary,_=apply_rotary_pos_emb(normed,normed,refs['cos'],refs['sin'])
      refs[p+name+'_rope']=rotary.transpose(1,2).contiguous()
     refs[p+'attn_output']=refs[p+'o_proj_input'];refs[p+'activation']=refs[p+'down_proj_input']
    for tag,value in state.items():
     if tag.endswith('_input'):continue
     rows.append(dict(prompt_length=length,step=step,tag=tag,kind='propagated',**stats(value,refs[tag])))
    for i,layer in enumerate(c.model.layers):
     p=f'layer.{i}.'
     for name in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
      module=getattr(layer.self_attn if name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp,name)
      actual=kernels.linear(refs[p+name+'_input'].contiguous(),module.weight)
      rows.append(dict(prompt_length=length,step=step,tag=p+name,kind='local_same_HF_input',**stats(actual,refs[p+name])))
     for name,module,input_tag in [('input_norm',layer.input_layernorm,'input_norm_input'),('post_norm',layer.post_attention_layernorm,'attn_residual')]:
      actual=im.rms_norm(refs[p+input_tag].contiguous(),module.weight,module.eps)
      rows.append(dict(prompt_length=length,step=step,tag=p+name,kind='local_same_HF_input',**stats(actual,refs[p+name])))
     actual=im.swiglu(refs[p+'gate_proj'],refs[p+'up_proj'])
     rows.append(dict(prompt_length=length,step=step,tag=p+'activation',kind='local_HF_gate_up',**stats(actual,refs[p+'activation'])))
     q=refs[p+'q_rope'];actual=kernels.attention(q,hc.key_cache[i].contiguous(),hc.value_cache[i].contiguous(),start).view_as(refs[p+'attn_output'])
     rows.append(dict(prompt_length=length,step=step,tag=p+'attn_output',kind='local_HF_QKV_cache',**stats(actual,refs[p+'attn_output'])))
    local_head=kernels.linear(refs['logits_input'].contiguous(),c.lm_head.weight)
    native_head=h.lm_head(state['final_norm'])
    rows.append(dict(prompt_length=length,step=step,tag='logits',kind='local_same_HF_input',**stats(local_head,y)))
    rows.append(dict(prompt_length=length,step=step,tag='logits',kind='native_head_on_candidate_hidden',**stats(native_head,y)))
    del x,y,z,state,local_head,native_head
 finally:
  for handle in handles:handle.remove()
 for p in files:assert sha(p)==hashes[str(p.relative_to(ROOT))],p
 result=dict(complete=True,diagnostic_only=True,production_changed=False,physical_GPU=1,GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,phases=phases,rows=rows,source_sha256=hashes,foreign_context_observations=list(foreign),resource_clean=not foreign,scope='Reference-only attribution at exact original B1S17/S65 and four cached decode inputs. Every wrapped public request reproduces unwrapped logits and initialized KV bitwise. Local operator probes consume exact HF operands. No qualification gate changed.')
 raw=json.dumps(result,indent=2)+'\n';assert len(raw.encode())<=4*2**20,'Bounded metrics cap exceeded';(out/'result.json').write_text(raw)
 print(json.dumps(dict(complete=True,phases=len(phases),rows=len(rows),resource_clean=not foreign)),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and out.is_relative_to(Path('/data1/tzh')) and not out.exists();out.mkdir(parents=True)
 foreign=[];stop=threading.Event()
 with stable_lease([1]):
  uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
  def monitor():
   while not stop.wait(.5):
    try:
     raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
     foreign.extend(line for line in raw.splitlines() if line.split(',')[0].strip()==uuid and int(line.split(',')[1])!=os.getpid())
    except Exception as exc:foreign.append(repr(exc))
  thread=threading.Thread(target=monitor,daemon=True);thread.start()
  try:run(a,out,foreign)
  except BaseException as exc:(out/'failure.json').write_text(json.dumps(dict(error=repr(exc),foreign_context_observations=foreign),indent=2)+'\n');raise
  finally:stop.set();thread.join()
if __name__=='__main__':main()
