"""Independent exact-HF-operand RoPE/activation/GEMM boundary diagnostics."""
import argparse,json,torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from qwen3_triton.inference import kernels
from qwen3_triton.kernels.math_precision import swiglu
from qwen3_triton.kernels.mlp_precision import gate_up_swiglu
from qwen3_triton.kernels.rotary import rotary_tables
from qwen3_triton.benchmark_utils.stable_idle import stable_lease

def stats(x,y):
 d=x.float()-y.float();return dict(max_abs=float(d.abs().max()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(y.float()).clamp_min(1e-12)),unequal=int((x!=y).sum()),numel=x.numel())
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2)
 with stable_lease([1]):
  torch.manual_seed(20261116);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  hf=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-8B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
  pos={};state={};rows=[];handles=[];phase=['prefill'];start=[0]
  def record(kind,layer,x,y):rows.append(dict(phase=phase[0],kind=kind,layer=layer,**stats(x,y)))
  def rotary(m,inp,out):
   pos['cos'],pos['sin']=out
   positions=kernels.positions(1,inp[1].shape[-1],start[0],torch.device('cuda'))
   c,s=rotary_tables(positions,m.inv_freq,torch.bfloat16)
   record('cos',-1,c,out[0]);record('sin',-1,s,out[1])
  handles.append(hf.model.rotary_emb.register_forward_hook(rotary))
  for i,layer in enumerate(hf.model.layers):
   def norm_hook(index):
    def hook(m,inp,out):
     x=inp[0];actual=kernels.norm_rope(x.contiguous(),m.weight,pos['cos'],pos['sin'],m.variance_epsilon)
     ref,_=apply_rotary_pos_emb(out.transpose(1,2),out.transpose(1,2),pos['cos'],pos['sin']);record('norm_rope',index,actual,ref.transpose(1,2))
    return hook
   handles.append(layer.self_attn.q_norm.register_forward_hook(norm_hook(i)));handles.append(layer.self_attn.k_norm.register_forward_hook(norm_hook(i)))
   def save(index,tag):
    def hook(m,inp,out):state[index,tag]=out
    return hook
   handles.append(layer.mlp.gate_proj.register_forward_hook(save(i,'gate')));handles.append(layer.mlp.up_proj.register_forward_hook(save(i,'up')))
   def down(index):
    def hook(m,inp):record('swiglu',index,swiglu(state[index,'gate'],state[index,'up']),inp[0])
    return hook
   handles.append(layer.mlp.down_proj.register_forward_pre_hook(down(i)))
   def mlp_hook(index):
    def hook(m,inp,out):
     x=inp[0].contiguous();y=gate_up_swiglu(x,m.gate_proj.weight,m.up_proj.weight,optimized=True)
     record('MLP_fused',index,kernels.linear(y,m.down_proj.weight),out)
     y=swiglu(kernels.linear(x,m.gate_proj.weight),kernels.linear(x,m.up_proj.weight))
     record('MLP_separate',index,kernels.linear(y,m.down_proj.weight),out)
    return hook
   handles.append(layer.mlp.register_forward_hook(mlp_hook(i)))
  torch.randint(1,100000,(1,21),device='cuda');ids=torch.randint(1,100000,(1,69),device='cuda');cache=DynamicCache()
  hf(ids[:,:65].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for j in range(4):phase[0]='decode_'+str(j);start[0]=65+j;hf(ids[:,65+j:66+j].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for h in handles:h.remove()
  result=dict(complete=True,diagnostic_only=True,rows=rows)
  Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
  summary={}
  for kind in set(x['kind'] for x in rows):
   r=[x for x in rows if x['kind']==kind];summary[kind]=dict(max_relative_l2=max(x['relative_l2'] for x in r),unequal=sum(x['unequal'] for x in r))
  print(json.dumps(dict(complete=True,summary=summary)),flush=True)
if __name__=='__main__':main()
