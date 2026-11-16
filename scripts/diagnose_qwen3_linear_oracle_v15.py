"""CPU FP64/fsum oracle for native/Triton BF16 linear disagreements."""
import argparse,json,math,torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from qwen3_triton.inference import kernels
from qwen3_triton.benchmark_utils.stable_idle import stable_lease

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2)
 with stable_lease([1]):
  torch.manual_seed(20261116);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  hf=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-8B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
  rows=[];phase=['prefill'];handles=[]
  def hook(name):
   def check(m,inp,ref):
    if phase[0]=='prefill':return
    x=inp[0].contiguous();y=kernels.linear(x,m.weight);accurate=kernels.linear(x,m.weight,accurate=True)
    xx=x.view(-1,x.shape[-1])[0].cpu().double();w=m.weight.cpu().double();oracle=w@xx
    r=ref.view(-1).cpu().float();v=y.view(-1).cpu().float();av=accurate.view(-1).cpu().float()
    diffs=(r!=v).nonzero().flatten().tolist();cases=[]
    for i in diffs:
     exact=math.fsum(a*b for a,b in zip(xx.tolist(),w[i].tolist()));h=float(r[i]);t=float(v[i]);mid=(h+t)/2
     cases.append(dict(index=i,HF=h,Triton=t,Triton_FP64_accum=float(av[i]),CPU_fsum=exact,midpoint=mid,midpoint_distance_in_ULP=(exact-mid)/abs(h-t),HF_closer=abs(exact-h)<abs(exact-t),Triton_closer=abs(exact-t)<abs(exact-h),ties=abs(exact-h)==abs(exact-t)))
    rows.append(dict(phase=phase[0],name=name,unequal=len(diffs),coordinates=cases,FP64_vs_fsum_max_abs=max((abs(float(oracle[c['index']])-c['CPU_fsum']) for c in cases),default=0)))
   return check
  selected=['model.layers.0.self_attn.q_proj','model.layers.12.mlp.down_proj','model.layers.24.self_attn.q_proj','model.layers.31.self_attn.q_proj','model.layers.33.mlp.down_proj']
  for n,m in hf.named_modules():
   if n in selected:handles.append(m.register_forward_hook(hook(n)))
  torch.randint(1,100000,(1,21),device='cuda');ids=torch.randint(1,100000,(1,69),device='cuda');cache=DynamicCache()
  hf(ids[:,:65].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for j in range(4):phase[0]='decode_'+str(j);hf(ids[:,65+j:66+j].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for h in handles:h.remove()
  result=dict(complete=True,diagnostic_only=True,rows=rows)
  Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
  cases=[c for r in rows for c in r['coordinates']]
  print(json.dumps(dict(complete=True,rows=len(rows),disagreements=len(cases),HF_closer=sum(c['HF_closer'] for c in cases),Triton_closer=sum(c['Triton_closer'] for c in cases),exact_ties=sum(c['ties'] for c in cases),max_midpoint_ULP=max((abs(c['midpoint_distance_in_ULP']) for c in cases),default=0))),flush=True)
if __name__=='__main__':main()
