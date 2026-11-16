"""Exact-HF-operand BF16 accumulation geometry probe; diagnostic only."""
import argparse,json,torch,triton
from pathlib import Path
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from qwen3_triton.inference import kernels
from qwen3_triton.kernels.linear import _gemm
from qwen3_triton.benchmark_utils.stable_idle import stable_lease

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2)
 with stable_lease([1]):
  torch.manual_seed(20261116);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  hf=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-8B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
  rows=[];phase=['prefill'];handles=[]
  def hook(name):
   def check(module,inputs,output):
    x=inputs[0].contiguous();flat=x.view(-1,x.shape[-1]);m,k=flat.shape;weight=module.weight;n=weight.shape[0]
    for bk in [16,32,64,128]:
     y=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
     _gemm[(triton.cdiv(m,16)*triton.cdiv(n,64),)](flat,weight.t(),y,m,n,k,k,1,1,k,16,64,bk,1,num_warps=4,num_stages=2)
     d=y.float()-output.view(m,n).float()
     rows.append(dict(phase=phase[0],name=name,K_tile=bk,M=m,N=n,K=k,unequal=int((y!=output.view(m,n)).sum()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(output.float()).clamp_min(1e-12))))
   return check
  for name,module in hf.named_modules():
   if isinstance(module,torch.nn.Linear) and any(name.startswith('model.layers.'+str(i)+'.') for i in [0,12,24,31,35]):handles.append(module.register_forward_hook(hook(name)))
  torch.randint(1,100000,(1,21),device='cuda');ids=torch.randint(1,100000,(1,69),device='cuda');cache=DynamicCache()
  hf(ids[:,:65].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for j in range(4):phase[0]='decode_'+str(j);hf(ids[:,65+j:66+j].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for h in handles:h.remove()
  result=dict(complete=True,diagnostic_only=True,rows=rows)
  Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(dict(complete=True,summary={bk:dict(unequal=sum(x['unequal'] for x in rows if x['K_tile']==bk),max_relative_l2=max(x['relative_l2'] for x in rows if x['K_tile']==bk)) for bk in [16,32,64,128]})),flush=True)
if __name__=='__main__':main()
