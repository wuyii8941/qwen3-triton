"""Compare Triton norms/linear on exact HF8B runtime operands; diagnostic only."""
import argparse,json,torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from qwen3_triton.inference import kernels
from qwen3_triton.benchmark_utils.stable_idle import stable_lease

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(2)
 with stable_lease([1]):
  torch.manual_seed(20261116);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  hf=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-8B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
  rows=[];phase=['prefill']
  def hook(name):
   def check(module,inputs,output):
    x=inputs[0]
    if module.__class__.__name__=='Qwen3RMSNorm':y=kernels.rms_norm(x.contiguous(),module.weight,module.variance_epsilon);kind='norm'
    elif isinstance(module,torch.nn.Linear):y=kernels.linear(x.contiguous(),module.weight,accurate=True);kind='linear'
    else:return
    d=y.float()-output.float()
    rows.append(dict(phase=phase[0],name=name,kind=kind,shape=list(x.shape),max_abs=float(d.abs().max()),mean_abs=float(d.abs().mean()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(output.float()).clamp_min(1e-12)),unequal=int((y!=output).sum()),numel=y.numel()))
   return check
  handles=[m.register_forward_hook(hook(n)) for n,m in hf.named_modules() if m.__class__.__name__=='Qwen3RMSNorm' or isinstance(m,torch.nn.Linear)]
  torch.randint(1,100000,(1,21),device='cuda');ids=torch.randint(1,100000,(1,69),device='cuda');cache=DynamicCache()
  hf(ids[:,:65].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for j in range(4):phase[0]='decode_'+str(j);hf(ids[:,65+j:66+j].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for h in handles:h.remove()
  result=dict(complete=True,diagnostic_only=True,rows=rows)
  Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(dict(complete=True,rows=len(rows),worst=sorted(rows,key=lambda x:x['relative_l2'],reverse=True)[:12])),flush=True)
if __name__=='__main__':main()
